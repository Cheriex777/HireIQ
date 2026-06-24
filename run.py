"""
run.py — HireIQ pipeline orchestrator
"""

import logging
import time
from pathlib import Path

# ── logging ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── project imports ──────────────────────────────────────────────────────────────
from src.config import (
    CANDIDATES_PATH,
    JOB_DESCRIPTION_PATH,
    OUTPUT_DIR,
    TOP_K,
    FINAL_TOP_K,
)
from src.semantic_retriever import SemanticRetriever
from src.feature_extractor import extract_features_batch
from src.final_ranker import rank_candidates as rank_candidates_fn
from src.reasoning_generator import generate_reasoning_batch
from src.submission_generator import generate_submission


# ── helpers ──────────────────────────────────────────────────────────────────────

def load_job_description(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    log.info("Loaded job description (%d chars)", len(text))
    return text


def parse_jd_skills(job_description: str) -> list[str]:
    """
    Reads 'Key skills:' line from job_description.txt.
    Falls back to curated list for the Redrob Senior AI Engineer JD.
    """
    for line in job_description.splitlines():
        if line.lower().startswith("key skills:"):
            skills = [s.strip() for s in line.split(":", 1)[1].split(",") if s.strip()]
            log.info("Parsed %d JD skills from 'Key skills:' line", len(skills))
            return skills

    fallback = [
        "python", "embeddings", "vector search", "hybrid retrieval",
        "ranking systems", "llms", "evaluation frameworks", "ndcg", "mrr",
        "sentence-transformers", "faiss", "pinecone", "elasticsearch",
        "learning-to-rank", "a/b testing", "nlp", "information retrieval",
        "fine-tuning llms", "rag", "recommendation systems", "search",
        "machine learning", "deep learning", "transformers",
    ]
    log.info("Using fallback JD skill list (%d skills)", len(fallback))
    return fallback


def flatten_retrieved(retrieved: list[dict]) -> tuple[list[dict], dict[str, float]]:
    """
    SemanticRetriever.retrieve() →
        [{candidate_id, semantic_score, candidate_data: {...}}, ...]

    Returns:
        candidates  — flat list of original candidate dicts (with candidate_id injected)
        sem_scores  — {candidate_id: semantic_score}
    """
    candidates: list[dict] = []
    sem_scores: dict[str, float] = {}
    for item in retrieved:
        cid = item["candidate_id"]
        candidate = dict(item["candidate_data"])
        candidate["candidate_id"] = cid
        candidates.append(candidate)
        sem_scores[cid] = item["semantic_score"]
    return candidates, sem_scores


def merge_features(candidates: list[dict], features: list[dict]) -> list[dict]:
    """
    Merge feature score dicts back onto full candidate dicts.
    rank_candidates() needs both raw data (profile, skills, etc.)
    AND score fields in the same dict.
    """
    feature_map: dict[str, dict] = {f["candidate_id"]: f for f in features}
    return [{**cand, **feature_map.get(cand.get("candidate_id", ""), {})}
            for cand in candidates]


# ── pipeline ─────────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    t0 = time.time()
    log.info("=== HireIQ pipeline start ===")

    # 1. Load JD ─────────────────────────────────────────────────────────────────
    job_description = load_job_description(JOB_DESCRIPTION_PATH)
    jd_skills: list[str] = parse_jd_skills(job_description)

    # 2. Semantic retrieval ───────────────────────────────────────────────────────
    log.info("Step 2/6 — Semantic retrieval (top_k=%d)", TOP_K)
    retriever = SemanticRetriever()
    retriever.load_and_encode_candidates(CANDIDATES_PATH)

    retrieved: list[dict] = retriever.retrieve(
        job_description=job_description,
        top_k=TOP_K,
    )
    log.info("Retrieved %d candidates", len(retrieved))

    # 3. Flatten ─────────────────────────────────────────────────────────────────
    candidates, sem_scores = flatten_retrieved(retrieved)

    # 4. Feature extraction ───────────────────────────────────────────────────────
    # extract_features_batch(candidates, jd_skills, semantic_scores) → list[dict]
    log.info("Step 3/6 — Feature extraction")
    features: list[dict] = extract_features_batch(
        candidates=candidates,
        jd_skills=jd_skills,
        semantic_scores=sem_scores,
    )
    log.info("Features extracted for %d candidates", len(features))

    # 5. Merge features onto candidate dicts ─────────────────────────────────────
    featured: list[dict] = merge_features(candidates, features)

    # 6. Hybrid ranking ───────────────────────────────────────────────────────────
    # rank_candidates(feature_rows, top_n) → list[dict] with final_score + rank
    log.info("Step 4/6 — Hybrid ranking → top %d", FINAL_TOP_K)
    ranked: list[dict] = rank_candidates_fn(
        feature_rows=featured,
        top_n=FINAL_TOP_K,
    )
    log.info(
        "Ranking complete. Top candidate id=%s  final_score=%.4f",
        ranked[0].get("candidate_id", "?"),
        ranked[0].get("final_score", 0.0),
    )

    # 7. Reasoning generation ─────────────────────────────────────────────────────
    # generate_reasoning_batch(candidates, ranked_features, jd_skills)
    #   → dict[candidate_id -> reasoning_string]
    log.info("Step 5/6 — Generating reasoning for %d candidates", len(ranked))
    reasoning_map: dict[str, str] = generate_reasoning_batch(
        candidates=candidates,       # raw records for profile/skills access
        ranked_features=ranked,      # scored + ranked dicts
        jd_skills=jd_skills,
    )

    # 8. Submission generation ────────────────────────────────────────────────────
    # generate_submission(ranked_candidates, reasoning_map, output_path) → path
    output_path = Path(OUTPUT_DIR) / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Step 6/6 — Writing submission → %s", output_path)

    generate_submission(
        ranked_candidates=ranked,
        reasoning_map=reasoning_map,
        output_path=str(output_path),
    )

    elapsed = time.time() - t0
    log.info("=== Pipeline complete in %.1fs ===", elapsed)
    log.info("Output → %s", output_path.resolve())


# ── entry point ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()