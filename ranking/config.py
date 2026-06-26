"""
Central configuration for the HireIQ ranking pipeline.

Every tunable lives here so final_ranker.py / evaluate.py never hardcode
magic numbers. Adjust weights and re-run — no need to touch scoring logic.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

RAW_CANDIDATES_PATH = os.path.join(DATA_DIR, "candidates.jsonl")

# Cached, flattened candidate corpus (text for embedding + structured record
# for rule-based scoring), built once by data_pipeline.py
CORPUS_CACHE_PATH = os.path.join(OUTPUT_DIR, "candidate_corpus.jsonl")

# Embedding artifacts, built once by embedding_engine.py
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "candidate_embeddings.npy")
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "candidate_index.faiss")
EMBEDDING_IDS_PATH = os.path.join(OUTPUT_DIR, "candidate_embedding_ids.json")

# Final outputs
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
EVAL_REPORT_PATH = os.path.join(OUTPUT_DIR, "evaluation_report.json")

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64  # lower to 32 if you hit memory pressure

# ---------------------------------------------------------------------------
# JD skill taxonomy
#
# Tiered, not flat: must-have skills count far more than nice-to-have ones.
# SKILL_SYNONYMS lets "LLMs" match "Large Language Models", etc. Candidate
# skill names get lowercased + synonym-canonicalized before matching, so the
# comparison isn't a brittle exact string match like the old scripts.
# ---------------------------------------------------------------------------

MUST_HAVE_SKILLS = [
    "nlp",
    "rag",
    "embeddings",
    "vector search",
    "information retrieval",
    "semantic search",
    "ranking systems",
]

NICE_TO_HAVE_SKILLS = [
    "faiss",
    "milvus",
    "recommendation systems",
    "fine-tuning llms",
    "llms",
]

# canonical_form -> list of raw strings that should map to it
SKILL_SYNONYMS = {
    "nlp": ["natural language processing", "nlp"],
    "rag": ["retrieval augmented generation", "retrieval-augmented generation", "rag"],
    "embeddings": ["embedding", "embeddings", "vector embeddings"],
    "vector search": ["vector search", "vector database", "vector db", "ann search",
                       "approximate nearest neighbor", "approximate nearest neighbour"],
    "information retrieval": ["information retrieval", "ir systems", "search retrieval"],
    "semantic search": ["semantic search"],
    "ranking systems": ["ranking systems", "learning to rank", "ltr", "ranking models"],
    "faiss": ["faiss"],
    "milvus": ["milvus"],
    "recommendation systems": ["recommendation systems", "recommender systems",
                                "recommendation engine", "recsys"],
    "fine-tuning llms": ["fine-tuning llms", "fine tuning llms", "llm fine-tuning",
                          "model fine-tuning", "peft", "lora"],
    "llms": ["llm", "llms", "large language model", "large language models"],
}

# Flatten synonyms into a single raw_string -> canonical lookup, built once.
_RAW_TO_CANONICAL = {}
for canonical, variants in SKILL_SYNONYMS.items():
    for v in variants:
        _RAW_TO_CANONICAL[v.lower().strip()] = canonical


def canonicalize_skill(raw_name: str) -> str:
    """Map a raw skill string to its canonical form, or lowercase-strip it
    unchanged if no synonym is registered."""
    if not raw_name:
        return ""
    key = raw_name.lower().strip()
    return _RAW_TO_CANONICAL.get(key, key)


# ---------------------------------------------------------------------------
# JD title taxonomy
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
# Experience requirement (override per-JD at runtime if needed)
# ---------------------------------------------------------------------------

DEFAULT_REQUIRED_YEARS = 5

# ---------------------------------------------------------------------------
# Hybrid ranking weights — must sum to 1.0 (validated in final_ranker.py)
# ---------------------------------------------------------------------------

WEIGHTS = {
    "semantic": 0.30,
    "skill": 0.30,
    "title": 0.15,
    "experience": 0.10,
    "behavioral": 0.15,
}

TOP_N = 100

# Stage 1 semantic shortlist size, per the official architecture spec
# (Stage 1: semantic retrieval -> top 1000 -> Stage 2: feature engineering
# only on that shortlist). Set this >= total corpus size to effectively
# disable the cutoff and score everyone on every component instead -- see
# the tradeoff note in final_ranker.py's HybridRanker.rank().
SHORTLIST_K = 1000
