"""HireIQ central configuration.

Single source of truth for paths, retrieval limits, ranking weights,
and shared constants used across the pipeline (run.py and src/*).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent.parent  # project root
CANDIDATES_PATH: str = str(BASE_DIR / "data" / "candidates.jsonl")
JOB_DESCRIPTION_PATH: str = str(BASE_DIR / "data" / "job_description.txt")
OUTPUT_DIR: str = str(BASE_DIR / "output")
SUBMISSION_PATH: str = str(BASE_DIR / "output" / "submission.csv")

# Semantic retrieval cache paths (previously hard-coded in semantic_retriever.py)
CANDIDATE_DOCS_PATH: str = str(BASE_DIR / "output" / "candidate_documents.json")
EMBEDDINGS_CACHE_PATH: str = str(BASE_DIR / "output" / "candidate_embeddings.npy")
IDS_CACHE_PATH: str = str(BASE_DIR / "output" / "candidate_ids.json")

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K: int = 500          # candidates passed from semantic retrieval -> ranker
FINAL_TOP_K: int = 100    # final submission size
SEMANTIC_MODEL_NAME: str = "all-MiniLM-L6-v2"  # previously hard-coded in semantic_retriever.py

# ── Ranking weights (single source of truth — must sum to ~1.0) ─────────────
# NOTE: These are the exact values previously hard-coded as DEFAULT_WEIGHTS
# inside src/final_ranker.py. The formula itself is unchanged; this dict is
# now imported by final_ranker.py instead of being redefined there.
RANKING_WEIGHTS: Dict[str, float] = {
    "semantic_score": 0.35,
    "skill_score": 0.27,
    "experience_score": 0.15,
    "behavior_score": 0.10,
    "title_score": 0.08,
    "profile_score": 0.05,
}

# Feature keys that must be present on every candidate row before ranking.
REQUIRED_FEATURE_KEYS: Tuple[str, ...] = (
    "candidate_id",
    "semantic_score",
    "skill_score",
    "experience_score",
    "behavior_score",
    "title_score",
)

# Maximum fraction by which the honeypot penalty may reduce a base score.
MAX_PENALTY_CAP: float = 0.9  # never reduce a score by more than 90%

# ── Honeypot detection thresholds ─────────────────────────────────────────────
# NOTE: previously hard-coded inside src/honeypot_detector.py. Values and
# detection logic are unchanged — only the constants moved here.
MAX_REALISTIC_SKILLS: int = 25            # beyond this, skill-padding suspected
MAX_REALISTIC_EXPERIENCE_YEARS: int = 30
MIN_AGE_FOR_EXPERIENCE: int = 18          # reserved: used if DOB/age becomes available; currently unused
SUSPICIOUS_SCORE_THRESHOLD: float = 0.5   # is_suspicious flag cutoff

# Titles implying high seniority that should correlate with some activity.
SENIOR_TITLE_KEYWORDS: List[str] = ["senior", "staff", "lead", "principal", "head", "director"]