"""
config.py — Central settings for HireIQ Ranking Engine
All tunable values live here. Change weights here, not in other files.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)  # one level up from ranking/

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
CANDIDATES_PATH = os.path.join(ROOT_DIR, "data", "sample_candidates.json")
JD_PATH         = os.path.join(ROOT_DIR, "data", "job_description.txt")
OUTPUT_DIR      = os.path.join(ROOT_DIR, "output")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
EMBEDDING_BATCH  = 32   # lower = less RAM used

# ---------------------------------------------------------------------------
# JD Skills — split into must-have and nice-to-have
# Proficiency weights: advanced=1.0, intermediate=0.6, beginner=0.3
# ---------------------------------------------------------------------------
MUST_HAVE_SKILLS = [
    "nlp", "embeddings", "vector search", "information retrieval",
    "semantic search", "ranking systems", "rag",
    "retrieval augmented generation",
]

NICE_TO_HAVE_SKILLS = [
    "faiss", "milvus", "pinecone", "elasticsearch", "weaviate", "qdrant",
    "recommendation systems", "fine-tuning llms", "lora", "qlora", "peft",
    "learning to rank", "sentence-transformers", "python",
    "hybrid retrieval", "a/b testing", "ndcg", "mrr",
]

PROFICIENCY_WEIGHTS = {
    "advanced":     1.0,
    "intermediate": 0.6,
    "beginner":     0.3,
}

# ---------------------------------------------------------------------------
# Important job titles
# ---------------------------------------------------------------------------
IMPORTANT_TITLES = [
    "Recommendation Systems Engineer",
    "Search Engineer",
    "Senior NLP Engineer",
    "Senior AI Engineer",
    "Lead AI Engineer",
    "Staff Machine Learning Engineer",
    "Machine Learning Engineer",
    "Applied ML Engineer",
    "AI Research Engineer",
    "Data Scientist",
]

# ---------------------------------------------------------------------------
# Experience requirement (years)
# ---------------------------------------------------------------------------
REQUIRED_YEARS = 5

# ---------------------------------------------------------------------------
# Stage 1: how many candidates pass semantic retrieval before scoring
# ---------------------------------------------------------------------------
SHORTLIST_K = 1000   # higher than 50, so all candidates pass Stage 1

# ---------------------------------------------------------------------------
# Hybrid scoring weights — must add up to 1.0
# ---------------------------------------------------------------------------
WEIGHTS = {
    "semantic":    0.30,
    "skill":       0.35,
    "title":       0.15,
    "experience":  0.10,
    "behavioral":  0.10,
}

TOP_N = 50  # set to 100 when you have the full 100k dataset
