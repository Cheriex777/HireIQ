"""
src/config.py — HireIQ central configuration
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────────
BASE_DIR             = Path(__file__).resolve().parent.parent  # project root
CANDIDATES_PATH = str(BASE_DIR / "data" / "sample_candidates.json")
JOB_DESCRIPTION_PATH = str(BASE_DIR / "data" / "job_description.txt")
OUTPUT_DIR           = str(BASE_DIR / "output")

# ── Retrieval ────────────────────────────────────────────────────────────────────
TOP_K       = 500   # candidates passed from semantic retrieval → ranker
FINAL_TOP_K = 100   # final submission size

# ── Scoring weights (must sum to 1.0) ───────────────────────────────────────────
WEIGHTS = {
    "semantic":    0.40,
    "skill":       0.25,
    "experience":  0.15,
    "behavior":    0.10,
    "title":       0.10,
}