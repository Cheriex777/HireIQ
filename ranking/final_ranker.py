"""
final_ranker.py
-----------------
The AI Ranking Engine. Given a Job Description, scores every candidate in
the pre-built FAISS index across five dimensions, normalizes each to a
comparable 0-1 scale (the old ranke_v1.py never did this -- its component
scores had wildly different ranges and silently let years_of_experience
dominate), combines them with configurable weights, and returns the top N.

Usage:
    from final_ranker import HybridRanker
    ranker = HybridRanker()
    results = ranker.rank(job_description_text, top_n=100)
    # results: list of dicts, sorted by final_score desc, each containing
    # candidate_id, final_score, component_scores, and the candidate record
    # (the record is what reasoning_generator.py consumes)
"""

import json
import re

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_minmax(values):
    """Min-max scale a list/array to 0-1. Computed at runtime over the
    actual candidate pool rather than hardcoded assumed ranges, since we
    don't know the true distribution of e.g. github_activity_score or
    search_appearance_30d ahead of time -- this adapts automatically and
    is robust to whatever scale the real data turns out to use."""
    arr = np.asarray(values, dtype="float64")
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _title_overlap_score(candidate_title, important_titles):
    """Token-overlap (Jaccard-style) match instead of exact string
    membership. 'Senior AI Engineer II' should still score well against
    'Senior AI Engineer' -- the old find_elite_candidates.py used exact
    `title in elite_titles`, which misses any real-world title variant."""
    if not candidate_title:
        return 0.0
    cand_tokens = set(re.findall(r"[a-z]+", candidate_title.lower()))
    if not cand_tokens:
        return 0.0

    best = 0.0
    for important in important_titles:
        imp_tokens = set(re.findall(r"[a-z]+", important.lower()))
        if not imp_tokens:
            continue
        overlap = len(cand_tokens & imp_tokens) / len(imp_tokens)
        best = max(best, overlap)
    return min(best, 1.0)


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

class SemanticScorer:
    """Cosine similarity between JD and every candidate document, via the
    pre-built FAISS index. O(n) query against the full corpus, which at
    100k x 384-dim is a sub-second operation -- no need to shortlist."""

    def __init__(self, index_path=None, ids_path=None, model_name=None):
        index_path = index_path or config.FAISS_INDEX_PATH
        ids_path = ids_path or config.EMBEDDING_IDS_PATH
        model_name = model_name or config.EMBEDDING_MODEL_NAME

        self.index = faiss.read_index(index_path)
        with open(ids_path, "r", encoding="utf-8") as f:
            self.candidate_ids = json.load(f)
        self.model = SentenceTransformer(model_name)

    def score(self, job_description: str):
        """Returns dict candidate_id -> cosine similarity (0-1, since both
        JD and candidate vectors are L2-normalized and FAISS IndexFlatIP
        gives inner product == cosine similarity)."""
        jd_vec = self.model.encode(
            [job_description], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        k = self.index.ntotal
        scores, indices = self.index.search(jd_vec, k)
        scores, indices = scores[0], indices[0]

        # cosine similarity is in [-1, 1]; clip negatives to 0 since a
        # negative match is meaningless for ranking purposes here
        result = {}
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            result[self.candidate_ids[idx]] = max(0.0, float(score))
        return result


class SkillScorer:
    """Weighted skill coverage: must-have skills count more than nice-to-have.
    Matches on canonicalized skill names (handles synonyms like 'LLM' vs
    'Large Language Model'), with a small bonus for skills mentioned in the
    candidate's headline/summary/career text even if absent from their
    formal skills list (catches incomplete skill-list data)."""

    def __init__(self, must_have=None, nice_to_have=None):
        self.must_have = [config.canonicalize_skill(s) for s in
                           (must_have or config.MUST_HAVE_SKILLS)]
        self.nice_to_have = [config.canonicalize_skill(s) for s in
                              (nice_to_have or config.NICE_TO_HAVE_SKILLS)]
        self.must_weight = 2.0
        self.nice_weight = 1.0
        self.max_possible = (len(self.must_have) * self.must_weight +
                              len(self.nice_to_have) * self.nice_weight)

    def score(self, record, full_text=""):
        candidate_skills = set(record.get("canonical_skills", []))
        total = 0.0
        matched = []

        for s in self.must_have:
            if s in candidate_skills:
                total += self.must_weight
                matched.append(s)
            elif s in full_text.lower():
                total += self.must_weight * 0.4  # partial credit, text-only mention
                matched.append(s + " (mentioned)")

        for s in self.nice_to_have:
            if s in candidate_skills:
                total += self.nice_weight
                matched.append(s)
            elif s in full_text.lower():
                total += self.nice_weight * 0.4
                matched.append(s + " (mentioned)")

        score = total / self.max_possible if self.max_possible else 0.0
        return min(score, 1.0), matched


class TitleScorer:
    """Current title weighted heavily; past titles in career_history give
    partial credit (a strong career trajectory toward this role matters)."""

    def __init__(self, important_titles=None):
        self.important_titles = important_titles or config.IMPORTANT_TITLES
        self.current_weight = 0.7
        self.history_weight = 0.3

    def score(self, record):
        current = _title_overlap_score(record.get("current_title", ""), self.important_titles)

        past_titles = record.get("past_titles", [])
        if past_titles:
            history_scores = [_title_overlap_score(t, self.important_titles) for t in past_titles]
            history = max(history_scores)
        else:
            history = 0.0

        return self.current_weight * current + self.history_weight * history


class ExperienceScorer:
    """Years of experience vs the JD's stated requirement. Meeting the bar
    matters more than exceeding it -- score plateaus instead of growing
    unbounded like the old code (which added raw years directly, letting a
    25-year generalist outscore a 5-year specialist purely on tenure)."""

    def __init__(self, required_years=None):
        self.required_years = required_years or config.DEFAULT_REQUIRED_YEARS

    def score(self, record):
        years = record.get("years_of_experience", 0) or 0
        req = self.required_years
        if req <= 0:
            return 1.0

        if years >= req:
            bonus = min((years - req) / req, 1.0)
            return min(1.0, 0.7 + 0.3 * bonus)
        return max(0.0, (years / req)) * 0.7


class BehavioralScorer:
    """Normalizes redrob_signals across the actual candidate pool at
    runtime (percentile/min-max), rather than assuming a fixed scale for
    fields whose true range (e.g. github_activity_score) isn't confirmed.
    Must be fit once over the full pool, then applied per-candidate."""

    RATE_FIELDS = [
        "recruiter_response_rate", "interview_completion_rate",
        "offer_acceptance_rate", "profile_completeness_score",
    ]
    COUNT_FIELDS = ["github_activity_score", "saved_by_recruiters_30d", "search_appearance_30d"]
    FLAG_FIELDS = ["open_to_work_flag", "verified_email", "verified_phone"]

    def __init__(self):
        self._fitted = False
        self._count_ranges = {}

    def fit(self, records):
        for field in self.COUNT_FIELDS:
            values = [r.get("redrob_signals", {}).get(field, 0) or 0 for r in records]
            arr = np.asarray(values, dtype="float64")
            self._count_ranges[field] = (arr.min(), arr.max())
        self._fitted = True

    def _normalize_count(self, field, value):
        lo, hi = self._count_ranges.get(field, (0, 1))
        if hi - lo < 1e-9:
            return 0.0
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

    def score(self, record):
        if not self._fitted:
            raise RuntimeError("BehavioralScorer.fit(all_records) must be called once before scoring")

        signals = record.get("redrob_signals", {})

        rate_score = np.mean([
            float(signals.get(f, 0) or 0) for f in self.RATE_FIELDS
        ])

        count_score = np.mean([
            self._normalize_count(f, signals.get(f, 0) or 0) for f in self.COUNT_FIELDS
        ])

        flag_score = np.mean([
            1.0 if signals.get(f) else 0.0 for f in self.FLAG_FIELDS
        ])

        # rates carry the most signal (they're already meaningful 0-1
        # behavioral metrics), counts and flags are supporting evidence
        return 0.5 * rate_score + 0.3 * count_score + 0.2 * flag_score


# ---------------------------------------------------------------------------
# Hybrid ranker
# ---------------------------------------------------------------------------

class HybridRanker:
    def __init__(self, corpus_path=None, weights=None, required_years=None,
                 must_have_skills=None, nice_to_have_skills=None, important_titles=None):
        self.corpus_path = corpus_path or config.CORPUS_CACHE_PATH
        self.weights = weights or config.WEIGHTS
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, "WEIGHTS must sum to 1.0"

        print("Loading candidate corpus...")
        self.corpus = self._load_corpus(self.corpus_path)

        print("Loading semantic search index...")
        self.semantic_scorer = SemanticScorer()

        self.skill_scorer = SkillScorer(must_have_skills, nice_to_have_skills)
        self.title_scorer = TitleScorer(important_titles)
        self.experience_scorer = ExperienceScorer(required_years)

        self.behavioral_scorer = BehavioralScorer()
        self.behavioral_scorer.fit([row["record"] for row in self.corpus.values()])

    @staticmethod
    def _load_corpus(path):
        corpus = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                corpus[row["candidate_id"]] = row
        return corpus

    def rank(self, job_description: str, top_n=None, shortlist_k=None):
        """
        Stage 1: semantic similarity over the full corpus, take top
                 `shortlist_k` candidates (default from config.SHORTLIST_K).
        Stage 2-3: skill / title / experience / behavioral scoring + hybrid
                 combination, run only on the shortlist.

        Tradeoff, stated plainly: this cutoff exists in the spec as a
        retrieve-then-rerank pattern, but since stages 2-3 here are cheap
        (no API calls, no heavy compute per candidate), it buys you no real
        speed -- only a recall risk, since a candidate with a perfect skill
        match but a generic document-text summary could be excluded before
        Stage 2 ever sees them. Set shortlist_k >= corpus size (e.g. to
        config-defined corpus length) to disable the cutoff and score every
        candidate on every component instead.
        """
        top_n = top_n or config.TOP_N
        shortlist_k = shortlist_k if shortlist_k is not None else config.SHORTLIST_K

        if shortlist_k < top_n:
            raise ValueError(
                f"shortlist_k ({shortlist_k}) must be >= top_n ({top_n}) -- "
                f"can't return {top_n} results from a shortlist of {shortlist_k}."
            )

        print("Stage 1: scoring semantic similarity for all candidates...")
        semantic_scores = self.semantic_scorer.score(job_description)

        shortlist_k = min(shortlist_k, len(self.corpus))
        shortlisted_ids = sorted(
            semantic_scores, key=semantic_scores.get, reverse=True
        )[:shortlist_k]
        print(f"Stage 1 shortlist: {len(shortlisted_ids)} of {len(self.corpus)} candidates "
              f"carried forward to Stage 2 (semantic score range "
              f"{semantic_scores[shortlisted_ids[-1]]:.3f} - {semantic_scores[shortlisted_ids[0]]:.3f}).")

        print("Stage 2/3: scoring skills / title / experience / behavior, then combining...")
        results = []
        for candidate_id in shortlisted_ids:
            row = self.corpus[candidate_id]
            record = row["record"]
            text = row["text"]

            sem = semantic_scores[candidate_id]
            skill, matched_skills = self.skill_scorer.score(record, full_text=text)
            title = self.title_scorer.score(record)
            experience = self.experience_scorer.score(record)
            behavioral = self.behavioral_scorer.score(record)

            final_score = (
                self.weights["semantic"] * sem +
                self.weights["skill"] * skill +
                self.weights["title"] * title +
                self.weights["experience"] * experience +
                self.weights["behavioral"] * behavioral
            )

            results.append({
                "candidate_id": candidate_id,
                "final_score": round(final_score * 100, 2),  # 0-100 scale for readability
                "component_scores": {
                    "semantic": round(sem, 4),
                    "skill": round(skill, 4),
                    "title": round(title, 4),
                    "experience": round(experience, 4),
                    "behavioral": round(behavioral, 4),
                },
                "matched_skills": matched_skills,
                "record": record,
            })

        results.sort(key=lambda r: r["final_score"], reverse=True)

        for rank, r in enumerate(results[:top_n], start=1):
            r["rank"] = rank

        return results[:top_n]


if __name__ == "__main__":
    # quick smoke test against a hardcoded JD; real usage goes through
    # run_pipeline.py with a JD file
    sample_jd = """
    Senior AI Engineer - Search & Recommendation
    Must have: NLP, RAG, Embeddings, Vector Search, FAISS, Milvus,
    Information Retrieval, Recommendation Systems, Semantic Search,
    Ranking Systems, Fine-tuning LLMs. 5+ years experience.
    """
    ranker = HybridRanker()
    top = ranker.rank(sample_jd, top_n=20)
    for r in top:
        print(r["rank"], r["final_score"], r["candidate_id"], r["component_scores"])
