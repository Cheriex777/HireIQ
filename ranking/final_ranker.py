"""
final_ranker.py — HireIQ AI Ranking Engine
Member 1 deliverable.

5 scoring dimensions:
  1. Semantic similarity  — how well the candidate's profile matches the JD
  2. Skill score          — JD skills matched by proficiency + assessment scores
  3. Title score          — current/past title vs JD target titles
  4. Experience score     — years vs JD requirement
  5. Behavioral score     — full redrob_signals (response rate, github, etc.)
"""

import json
import os
import re
import sys

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ===========================================================================
# STEP 1 — Load candidates
# ===========================================================================

def load_candidates(path=None):
    path = path or config.CANDIDATES_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} candidates from {path}")
    return data


# ===========================================================================
# STEP 2 — Build text document for embedding
# ===========================================================================

def build_document(candidate):
    profile = candidate.get("profile") or {}
    skills  = candidate.get("skills") or []
    career  = candidate.get("career_history") or []

    title    = profile.get("current_title", "")
    headline = profile.get("headline", "")
    summary  = profile.get("summary", "")

    skill_names = [s["name"] for s in skills if isinstance(s, dict) and s.get("name")]

    job_texts = []
    for job in career:
        if isinstance(job, dict):
            t = job.get("title", "")
            d = job.get("description", "")
            if t or d:
                job_texts.append(f"{t}. {d}".strip())

    parts = [
        f"Title: {title}",
        f"Headline: {headline}",
        f"Summary: {summary}",
        f"Skills: {', '.join(skill_names)}",
    ] + job_texts

    return "\n".join(p for p in parts if p.strip())


# ===========================================================================
# STEP 3 — Semantic scorer
# ===========================================================================

def semantic_score_all(jd_text, candidates):
    print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    print("Embedding job description...")
    jd_vec = model.encode([jd_text], batch_size=1, show_progress_bar=False)

    print(f"Embedding {len(candidates)} candidate profiles...")
    docs = [build_document(c) for c in candidates]
    candidate_vecs = model.encode(
        docs,
        batch_size=config.EMBEDDING_BATCH,
        show_progress_bar=True,
    )

    sims = cosine_similarity(jd_vec, candidate_vecs)[0]
    return {
        c["candidate_id"]: float(max(0.0, sims[i]))
        for i, c in enumerate(candidates)
    }


# ===========================================================================
# STEP 4 — Skill scorer
# Uses: proficiency level + endorsements + duration + skill_assessment_scores
# ===========================================================================

def skill_score(candidate):
    """
    Three layers of evidence per skill:
      Layer 1 — Proficiency (self-reported): advanced=1.0, intermediate=0.6, beginner=0.3
      Layer 2 — Endorsements + duration (social/time proof)
      Layer 3 — skill_assessment_scores (objective test score 0-100)

    Layer 3 overrides layers 1+2 when available because it's objective,
    not self-reported.
    """
    skills   = candidate.get("skills") or []
    signals  = candidate.get("redrob_signals") or {}

    # skill_assessment_scores: {"NLP": 38.8, "Fine-tuning LLMs": 41.6, ...}
    # These are objective test scores out of 100
    assessment_scores = signals.get("skill_assessment_scores") or {}
    # Normalize assessment keys to lowercase for matching
    assessments_lower = {k.lower(): v for k, v in assessment_scores.items()}

    # Build candidate skill lookup: lowercase name -> full skill dict
    candidate_skills = {}
    for s in skills:
        if not isinstance(s, dict):
            continue
        name = (s.get("name") or "").lower().strip()
        if name:
            candidate_skills[name] = s

    total_score  = 0.0
    max_possible = 0.0
    matched      = []

    all_jd_skills = (
        [(s, 2.0) for s in config.MUST_HAVE_SKILLS] +
        [(s, 1.0) for s in config.NICE_TO_HAVE_SKILLS]
    )

    for jd_skill, weight in all_jd_skills:
        max_possible += weight

        # --- Find this JD skill in candidate's skill list ---
        match = None
        for cname, cskill in candidate_skills.items():
            if jd_skill in cname or cname in jd_skill:
                match = cskill
                break

        if not match:
            continue

        skill_display_name = match.get("name", jd_skill)

        # --- Layer 1: proficiency ---
        proficiency = (match.get("proficiency") or "beginner").lower()
        prof_weight = config.PROFICIENCY_WEIGHTS.get(proficiency, 0.3)

        # --- Layer 2: endorsements + duration bonus ---
        endorsements  = min(match.get("endorsements", 0) or 0, 50)
        endorse_bonus = (endorsements / 50) * 0.15

        duration      = min(match.get("duration_months", 0) or 0, 36)
        duration_bonus = (duration / 36) * 0.10

        combined = min(prof_weight + endorse_bonus + duration_bonus, 1.0)

        # --- Layer 3: objective assessment score (overrides if available) ---
        # Check if there's a test score for this skill
        assessment_val = None
        for akey, aval in assessments_lower.items():
            if jd_skill in akey or akey in jd_skill:
                assessment_val = aval
                break

        if assessment_val is not None:
            # Assessment score 0-100. Weight it 60% assessment, 40% profile signals
            assessment_normalized = assessment_val / 100.0
            combined = 0.60 * assessment_normalized + 0.40 * combined
            matched.append(f"{skill_display_name} ({proficiency}, assessed: {assessment_val:.0f}/100)")
        else:
            matched.append(f"{skill_display_name} ({proficiency})")

        total_score += weight * combined

    score = total_score / max_possible if max_possible > 0 else 0.0
    return min(score, 1.0), matched


# ===========================================================================
# STEP 5 — Title scorer
# ===========================================================================

def title_score(candidate):
    def overlap(title_a, title_b):
        if not title_a or not title_b:
            return 0.0
        tokens_a = set(re.findall(r"[a-z]+", title_a.lower()))
        tokens_b = set(re.findall(r"[a-z]+", title_b.lower()))
        if not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_b)

    profile       = candidate.get("profile") or {}
    current_title = profile.get("current_title", "") or ""
    career        = candidate.get("career_history") or []
    past_titles   = [j.get("title", "") for j in career if isinstance(j, dict)]

    best_current = max(
        (overlap(current_title, t) for t in config.IMPORTANT_TITLES),
        default=0.0
    )
    best_past = max(
        (overlap(pt, t) for pt in past_titles for t in config.IMPORTANT_TITLES),
        default=0.0
    ) if past_titles else 0.0

    return min(0.7 * best_current + 0.3 * best_past, 1.0)


# ===========================================================================
# STEP 6 — Experience scorer
# ===========================================================================

def experience_score(candidate):
    profile = candidate.get("profile") or {}
    years   = float(profile.get("years_of_experience", 0) or 0)
    req     = config.REQUIRED_YEARS

    if years >= req:
        bonus = min((years - req) / req, 1.0)
        return min(0.7 + 0.3 * bonus, 1.0)
    return max(0.0, (years / req) * 0.7)


# ===========================================================================
# STEP 7 — Behavioral scorer
# Uses the FULL real redrob_signals schema
# ===========================================================================

def _pool_max(candidates, key):
    """Get max value of a signal across the whole pool (for normalization)."""
    vals = []
    for c in candidates:
        v = (c.get("redrob_signals") or {}).get(key)
        if v is not None and v != -1:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    return max(vals) if vals else 1.0


def behavioral_score(candidate, all_candidates, pool_maxes=None):
    """
    Scores based on the full redrob_signals schema.
    pool_maxes is pre-computed once for efficiency.
    """
    signals = candidate.get("redrob_signals") or {}

    def get(key, default=0.0):
        v = signals.get(key, default)
        if v is None or v == -1:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def norm(key):
        """Normalize a count field against pool max."""
        val = get(key)
        mx  = pool_maxes.get(key, 1.0) if pool_maxes else _pool_max(all_candidates, key)
        return val / mx if mx > 0 else 0.0

    # --- Rate signals (already 0-1) ---
    recruiter_response  = get("recruiter_response_rate")       # 0.34
    interview_complete  = get("interview_completion_rate")     # 0.71
    offer_acceptance    = max(0.0, get("offer_acceptance_rate"))  # 0.58

    # --- Profile quality ---
    completeness = get("profile_completeness_score") / 100.0   # 86.9 -> 0.869

    # --- Activity signals (normalize against pool) ---
    github_norm   = norm("github_activity_score")              # 9.2
    search_norm   = norm("search_appearance_30d")              # 249
    saved_norm    = norm("saved_by_recruiters_30d")            # 4
    views_norm    = norm("profile_views_received_30d")         # 23
    connections   = min(get("connection_count") / 500.0, 1.0) # 356 -> 0.71

    # --- Responsiveness (lower hours = better, invert) ---
    avg_hours = get("avg_response_time_hours", 999)
    responsiveness = max(0.0, 1.0 - (avg_hours / 240.0))      # 240hrs = 0, 0hrs = 1

    # --- Boolean signals ---
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.0
    verified     = 1.0 if (signals.get("verified_email") and signals.get("verified_phone")) else 0.0
    linkedin     = 1.0 if signals.get("linkedin_connected") else 0.5

    # --- Combine ---
    # Weights reflect what actually predicts a candidate will engage:
    # response rates + interview completion are the strongest signals
    score = (
        0.20 * recruiter_response  +
        0.15 * interview_complete  +
        0.10 * offer_acceptance    +
        0.10 * completeness        +
        0.10 * github_norm         +
        0.08 * search_norm         +
        0.08 * saved_norm          +
        0.05 * views_norm          +
        0.05 * connections         +
        0.04 * responsiveness      +
        0.03 * open_to_work        +
        0.01 * verified            +
        0.01 * linkedin
    )
    return min(score, 1.0)


# ===========================================================================
# STEP 8 — Hybrid ranking
# ===========================================================================

def rank_candidates(jd_text, candidates, top_n=None):
    top_n = top_n or config.TOP_N
    w     = config.WEIGHTS

    # Stage 1: semantic similarity
    semantic_scores = semantic_score_all(jd_text, candidates)

    # Shortlist top K by semantic score
    shortlist_k = min(config.SHORTLIST_K, len(candidates))
    shortlisted = sorted(
        candidates,
        key=lambda c: semantic_scores.get(c["candidate_id"], 0),
        reverse=True
    )[:shortlist_k]

    print(f"Stage 1 shortlist: {len(shortlisted)} candidates")
    print("Stage 2/3: scoring skills, title, experience, behavior...")

    # Pre-compute pool maxes once (not inside the loop)
    count_fields = [
        "github_activity_score", "search_appearance_30d",
        "saved_by_recruiters_30d", "profile_views_received_30d",
        "connection_count",
    ]
    pool_maxes = {f: _pool_max(candidates, f) for f in count_fields}

    results = []
    for c in shortlisted:
        cid = c["candidate_id"]

        sem            = semantic_scores.get(cid, 0.0)
        sk, matched    = skill_score(c)
        ti             = title_score(c)
        ex             = experience_score(c)
        beh            = behavioral_score(c, candidates, pool_maxes=pool_maxes)

        final = (
            w["semantic"]   * sem +
            w["skill"]      * sk  +
            w["title"]      * ti  +
            w["experience"] * ex  +
            w["behavioral"] * beh
        )

        results.append({
            "candidate_id":    cid,
            "final_score":     round(final * 100, 2),
            "component_scores": {
                "semantic":    round(sem, 4),
                "skill":       round(sk,  4),
                "title":       round(ti,  4),
                "experience":  round(ex,  4),
                "behavioral":  round(beh, 4),
            },
            "matched_skills": matched,
            "candidate":      c,
        })

    results.sort(key=lambda r: r["final_score"], reverse=True)
    for rank, r in enumerate(results[:top_n], start=1):
        r["rank"] = rank

    return results[:top_n]


# ===========================================================================
# Quick test — run this file directly
# ===========================================================================

if __name__ == "__main__":
    candidates = load_candidates()
    jd_text    = open(config.JD_PATH, encoding="utf-8").read()
    top        = rank_candidates(jd_text, candidates, top_n=config.TOP_N)

    print(f"\n{'Rank':<5} {'Score':<8} {'ID':<15} {'Title'}")
    print("-" * 65)
    for r in top[:10]:
        profile = r["candidate"]["profile"]
        title   = profile.get("current_title", "Unknown")[:30]
        print(f"{r['rank']:<5} {r['final_score']:<8} {r['candidate_id']:<15} {title}")
