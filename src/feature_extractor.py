"""
feature_extractor.py
HireIQ - Feature Extraction Module

Extracts skill, experience, title, behavior, and profile-completeness
features for candidates given a job description context.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hireiq.feature_extractor")
logging.basicConfig(level=logging.INFO)


# ----------------------------------------------------------------------
# Static reference data
# ----------------------------------------------------------------------

TITLE_SCORE_MAP: Dict[str, float] = {
    # Perfect fit
    "senior ai engineer":                   100,
    "lead ai engineer":                     100,
    "staff ai engineer":                    100,
    "senior machine learning engineer":     95,
    "lead machine learning engineer":       95,
    "staff machine learning engineer":      95,
    "senior nlp engineer":                  95,
    "senior search engineer":               95,
    "senior recommendation engineer":       95,
    # Strong fit
    "machine learning engineer":            85,
    "ai engineer":                          85,
    "nlp engineer":                         85,
    "search engineer":                      85,
    "recommendation systems engineer":      85,
    "applied scientist":                    80,
    "research engineer":                    75,
    # Moderate fit
    "data scientist":                       70,
    "senior data scientist":                75,
    "lead data scientist":                  78,
    "ml platform engineer":                 72,
    "software engineer":                    55,
    "senior software engineer":             58,
    "backend engineer":                     45,
    "senior backend engineer":              48,
    "data engineer":                        40,
    "analytics engineer":                   35,
    # Weak / no fit
    "frontend engineer":                    20,
    "devops engineer":                      20,
    "product manager":                      10,
    "hr manager":                            0,
    "marketing manager":                     0,
    "sales manager":                         0,
}

# Partial title keywords for fuzzy matching (checked if exact match fails)
TITLE_KEYWORD_SCORES: Dict[str, float] = {
    "machine learning":  85,
    "ml engineer":       85,
    "ai engineer":       85,
    "nlp":               82,
    "search":            80,
    "ranking":           80,
    "retrieval":         80,
    "recommendation":    78,
    "data scientist":    70,
    "applied scientist": 75,
    "research engineer": 72,
    "software engineer": 55,
    "backend":           45,
    "data engineer":     40,
    "frontend":          20,
    "devops":            20,
}

DEFAULT_TITLE_SCORE = 45.0

SKILL_SYNONYMS: Dict[str, str] = {
    "tf":                           "tensorflow",
    "tensor flow":                  "tensorflow",
    "vector db":                    "vector search",
    "vector databases":             "vector search",
    "vector database":              "vector search",
    "information retrieval":        "information retrieval",
    "ir":                           "information retrieval",
    "llm finetuning":               "fine-tuning llms",
    "llm fine-tuning":              "fine-tuning llms",
    "finetuning":                   "fine-tuning llms",
    "fine tuning":                  "fine-tuning llms",
    "rag pipelines":                "rag",
    "retrieval augmented generation": "rag",
    "retrieval-augmented generation": "rag",
    "nlp":                          "nlp",
    "natural language processing":  "nlp",
    "sentence transformers":        "sentence-transformers",
    "sbert":                        "sentence-transformers",
    "elastic search":               "elasticsearch",
    "open search":                  "elasticsearch",
    "opensearch":                   "elasticsearch",
    "xgboost":                      "learning-to-rank",
    "lightgbm":                     "learning-to-rank",
    "ltr":                          "learning-to-rank",
    "a/b test":                     "a/b testing",
    "ab testing":                   "a/b testing",
    "hybrid search":                "hybrid retrieval",
    "dense retrieval":              "hybrid retrieval",
    "sentence transformers":    "sentence-transformers",
    "hugging face transformers": "sentence-transformers",
    "huggingface transformers": "sentence-transformers",
    "faiss":                    "faiss",
    "embeddings":               "embeddings",
    "machine learning":         "machine learning",
    "feature engineering":      "feature engineering",
    "information retrieval":    "information retrieval",
    "mlops":                    "mlops",
    "mlflow":                   "mlflow",
    "scikit-learn":             "scikit-learn",
    "scikit learn":             "scikit-learn",
    "sentence transformers":     "sentence-transformers",
    "hugging face transformers": "sentence-transformers",
    "huggingface transformers": "sentence-transformers",
    "llms":                      "llms",
    "large language models":     "llms",
    "a/b testing":               "a/b testing",
    "ab testing":                "a/b testing",
    "learning to rank":          "learning-to-rank",
    "vector search":             "vector search",
    "hybrid retrieval":          "hybrid retrieval",
    "ranking systems":           "ranking systems",
    "evaluation frameworks":     "evaluation frameworks",
}

BEHAVIOR_FIELDS: List[str] = [
    "recruiter_response_rate",
    "github_activity_score",
    "search_appearance_30d",
    "saved_by_recruiters_30d",
    "interview_completion_rate",
    "offer_acceptance_rate",
]

BEHAVIOR_WEIGHTS: Dict[str, float] = {
    "recruiter_response_rate":   0.20,
    "github_activity_score":     0.15,
    "search_appearance_30d":     0.15,
    "saved_by_recruiters_30d":   0.20,
    "interview_completion_rate": 0.20,
    "offer_acceptance_rate":     0.10,
}

# These fields are 0-1 fractions in schema → scale to 0-100
BEHAVIOR_FRACTION_FIELDS = {
    "recruiter_response_rate",
    "interview_completion_rate",
    "offer_acceptance_rate",
}

# search_appearance_30d and saved_by_recruiters_30d are raw counts → normalize
BEHAVIOR_COUNT_CAPS: Dict[str, float] = {
    "search_appearance_30d":  500.0,   # cap at 500 appearances
    "saved_by_recruiters_30d": 50.0,   # cap at 50 saves
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _normalize_skill(skill: str) -> str:
    s = skill.strip().lower()
    s_no_hyphen = s.replace("-", " ")
    s_with_hyphen = s.replace(" ", "-")
    return SKILL_SYNONYMS.get(s,
           SKILL_SYNONYMS.get(s_no_hyphen,
           SKILL_SYNONYMS.get(s_with_hyphen, s)))


def _normalize_skill_set(skills: Optional[List[Any]]) -> set:
    if not skills:
        return set()
    try:
        names = []
        for s in skills:
            if isinstance(s, dict):
                name = s.get("name", "")
            elif isinstance(s, str):
                name = s
            else:
                continue
            if name and name.strip():
                names.append(_normalize_skill(name))
        return set(names)
    except Exception as exc:
        logger.warning("Failed to normalize skill list %s: %s", skills, exc)
        return set()

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------------
# Scoring functions
# ----------------------------------------------------------------------

def calculate_skill_score(jd_skills: List[str], candidate_skills: List[str]) -> float:
    """
    Percentage overlap of normalized JD skills found in candidate skills.
    Also gives partial credit for skills found via substring matching.
    """
    try:
        jd_set = _normalize_skill_set(jd_skills)
        cand_set = _normalize_skill_set(candidate_skills)

        if not jd_set:
            logger.warning("Empty JD skill set; returning 0 skill score.")
            return 0.0

        # Exact overlap
        exact_overlap = jd_set & cand_set

        # Partial credit: JD skill appears as substring in any candidate skill
        partial_matches = set()
        for jd_skill in jd_set - exact_overlap:
            for cand_skill in cand_set:
                if jd_skill in cand_skill or cand_skill in jd_skill:
                    partial_matches.add(jd_skill)
                    break

        score = (len(exact_overlap) + 0.5 * len(partial_matches)) / len(jd_set) * 100.0
        return round(min(max(score, 0.0), 100.0), 2)
    except Exception as exc:
        logger.error("calculate_skill_score failed: %s", exc)
        return 0.0


def calculate_experience_score(years_of_experience: Any) -> float:
    """
    Score years of experience against JD sweet spot of 5-9 years.
    Peak score at 6-8 years; penalizes too little or too much.
    """
    try:
        years = _safe_float(years_of_experience, default=0.0)
        years = max(years, 0.0)

        if years < 2:
            return 10.0
        elif years < 4:
            return 35.0
        elif years < 5:
            return 60.0
        elif years <= 9:
            # Sweet spot: linear peak at 6-8
            if years <= 6:
                return round(60.0 + (years - 5) * 20.0, 2)   # 60→80
            elif years <= 8:
                return 100.0                                    # peak
            else:
                return round(100.0 - (years - 8) * 10.0, 2)   # 100→80
        elif years <= 12:
            return round(80.0 - (years - 9) * 5.0, 2)         # 80→65
        else:
            return 60.0
    except Exception as exc:
        logger.error("calculate_experience_score failed: %s", exc)
        return 0.0


def calculate_title_score(current_title: Optional[str]) -> float:
    """
    Map candidate's current title to a relevance score.
    Tries exact match first, then keyword-based fuzzy match.
    """
    try:
        if not current_title or not isinstance(current_title, str):
            return DEFAULT_TITLE_SCORE

        key = current_title.strip().lower()

        # Exact match
        if key in TITLE_SCORE_MAP:
            return float(TITLE_SCORE_MAP[key])

        # Keyword fuzzy match
        best_score = DEFAULT_TITLE_SCORE
        for keyword, score in TITLE_KEYWORD_SCORES.items():
            if keyword in key:
                best_score = max(best_score, score)

        return best_score
    except Exception as exc:
        logger.error("calculate_title_score failed: %s", exc)
        return DEFAULT_TITLE_SCORE


def calculate_behavior_score(redrob_signals: Optional[Dict[str, Any]]) -> float:
    """
    Weighted normalized score from behavioral / engagement signals.
    Handles fraction fields (0-1), count fields, and -1 sentinels.
    Also applies open_to_work and notice_period bonuses.
    """
    try:
        if not redrob_signals or not isinstance(redrob_signals, dict):
            return 0.0

        total = 0.0
        weight_sum = 0.0

        for field in BEHAVIOR_FIELDS:
            raw_val = redrob_signals.get(field, 0.0)
            val = _safe_float(raw_val, default=0.0)

            # -1 sentinel = no data → treat as 0
            if val < 0:
                val = 0.0

            # Scale fraction fields (0-1) to 0-100
            if field in BEHAVIOR_FRACTION_FIELDS:
                val = val * 100.0

            # Normalize count fields to 0-100 using known caps
            elif field in BEHAVIOR_COUNT_CAPS:
                cap = BEHAVIOR_COUNT_CAPS[field]
                val = min(val / cap * 100.0, 100.0)

            val = min(max(val, 0.0), 100.0)
            weight = BEHAVIOR_WEIGHTS.get(field, 0.0)
            total += val * weight
            weight_sum += weight

        if weight_sum == 0:
            return 0.0

        base_score = total / weight_sum

        # Bonus: open to work (+5 points)
        if redrob_signals.get("open_to_work_flag", False):
            base_score = min(base_score + 5.0, 100.0)

        # Bonus: short notice period ≤30 days (+3 points)
        notice = _safe_float(redrob_signals.get("notice_period_days"), default=90.0)
        if 0 <= notice <= 30:
            base_score = min(base_score + 3.0, 100.0)

        return round(base_score, 2)
    except Exception as exc:
        logger.error("calculate_behavior_score failed: %s", exc)
        return 0.0


def calculate_profile_completeness_score(
    profile: Dict[str, Any],
    career_history: Optional[List[Any]] = None,
    education: Optional[List[Any]] = None,
    skills: Optional[List[Any]] = None,
) -> float:
    """Score based on how complete a candidate's profile data is."""
    try:
        checks = [
            bool(profile.get("headline")),
            bool(profile.get("summary")),
            profile.get("years_of_experience") is not None,
            bool(profile.get("current_title")),
            bool(profile.get("current_company")),
            bool(career_history),
            bool(education),
            bool(skills),
        ]
        return round(100.0 * sum(checks) / len(checks), 2)
    except Exception as exc:
        logger.error("calculate_profile_completeness_score failed: %s", exc)
        return 0.0


# ----------------------------------------------------------------------
# Aggregator
# ----------------------------------------------------------------------

def extract_all_features(
    candidate: Dict[str, Any],
    jd_skills: List[str],
    semantic_score: float,
) -> Dict[str, Any]:
    """
    Build full feature dict for one candidate.

    Args:
        candidate: raw candidate record.
        jd_skills: list of skills extracted from the job description.
        semantic_score: precomputed semantic similarity score (0-100).

    Returns:
        dict with candidate_id + all score fields.
    """
    candidate_id = candidate.get("candidate_id", "UNKNOWN")

    try:
        profile        = candidate.get("profile", {}) or {}
        career_history = candidate.get("career_history", []) or []
        education      = candidate.get("education", []) or []
        redrob_signals = candidate.get("redrob_signals", {}) or {}

        # skills is list of {name, proficiency, endorsements} objects
        raw_skills = candidate.get("skills", []) or []
        skills = [
            s["name"] for s in raw_skills
            if isinstance(s, dict) and s.get("name")
        ]

        skill_score      = calculate_skill_score(jd_skills, skills)
        experience_score = calculate_experience_score(profile.get("years_of_experience"))
        title_score      = calculate_title_score(profile.get("current_title"))
        behavior_score   = calculate_behavior_score(redrob_signals)
        profile_score    = calculate_profile_completeness_score(
            profile, career_history, education, skills
        )

        sem_score = min(max(_safe_float(semantic_score, 0.0), 0.0), 100.0)

        return {
            "candidate_id":      candidate_id,
            "semantic_score":    round(sem_score, 2),
            "skill_score":       skill_score,
            "experience_score":  experience_score,
            "behavior_score":    behavior_score,
            "title_score":       title_score,
            "profile_score":     profile_score,
        }
    except Exception as exc:
        logger.error("extract_all_features failed for %s: %s", candidate_id, exc)
        return {
            "candidate_id":      candidate_id,
            "semantic_score":    0.0,
            "skill_score":       0.0,
            "experience_score":  0.0,
            "behavior_score":    0.0,
            "title_score":       0.0,
            "profile_score":     0.0,
        }


def extract_features_batch(
    candidates: List[Dict[str, Any]],
    jd_skills: List[str],
    semantic_scores: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Run extract_all_features over a list of candidates.

    Args:
        candidates: list of raw candidate records.
        jd_skills: skills parsed from job description.
        semantic_scores: dict mapping candidate_id -> semantic_score.

    Returns:
        list of feature dicts.
    """
    results: List[Dict[str, Any]] = []
    for candidate in candidates:
        cid = candidate.get("candidate_id", "UNKNOWN")
        sem_score = semantic_scores.get(cid, 0.0)
        try:
            features = extract_all_features(candidate, jd_skills, sem_score)
            results.append(features)
        except Exception as exc:
            logger.error("Skipping candidate %s: %s", cid, exc)

    logger.info("Extracted features for %d/%d candidates.", len(results), len(candidates))
    return results