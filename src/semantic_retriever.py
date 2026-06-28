"""
semantic_retriever.py - Generates candidate embeddings and retrieves
top-N candidates semantically closest to a job description.

Usage:
    from src.semantic_retriever import SemanticRetriever

    retriever = SemanticRetriever()
    retriever.load_and_encode_candidates("data/candidates.jsonl")
    results = retriever.retrieve(job_description, top_k=500)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    CANDIDATE_DOCS_PATH,
    EMBEDDINGS_CACHE_PATH,
    IDS_CACHE_PATH,
    OUTPUT_DIR,
    SEMANTIC_MODEL_NAME,
)
from src.utils import build_candidate_document, load_jsonl, save_json

logger = logging.getLogger("hireiq.semantic_retriever")
logging.basicConfig(level=logging.INFO)


class SemanticRetriever:
    """
    Encodes candidate profiles using SentenceTransformer and performs
    cosine-similarity-based retrieval against a job description.
    """

    def __init__(self, model_name: str = SEMANTIC_MODEL_NAME) -> None:
        """Load the SentenceTransformer model and initialize empty state.

        Args:
            model_name: Name of the SentenceTransformer model to load.
                Defaults to ``config.SEMANTIC_MODEL_NAME``.
        """
        logger.info("Loading model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.candidate_ids: List[str] = []
        self.candidate_embeddings: Optional[np.ndarray] = None
        self.candidates_raw: List[Dict] = []

    # ------------------------------------------------------------------
    # Building / caching embeddings
    # ------------------------------------------------------------------

    def load_and_encode_candidates(
        self,
        jsonl_path: str,
        force_recompute: bool = False,
    ) -> None:
        """
        Load candidates from .jsonl, build text documents, encode them.
        Caches embeddings to disk so re-runs are fast.

        Args:
            jsonl_path: Path to the candidates .jsonl file.
            force_recompute: If True, ignore any existing cache and
                re-encode all candidates from scratch.
        """
        if (
            not force_recompute
            and os.path.exists(EMBEDDINGS_CACHE_PATH)
            and os.path.exists(IDS_CACHE_PATH)
        ):
            logger.info("Loading cached embeddings...")
            self.candidate_embeddings = np.load(EMBEDDINGS_CACHE_PATH)
            with open(IDS_CACHE_PATH, "r") as f:
                self.candidate_ids = json.load(f)
            self.candidates_raw = load_jsonl(jsonl_path)
            logger.info(
                "Loaded %d candidates with embeddings shape %s",
                len(self.candidate_ids),
                self.candidate_embeddings.shape,
            )
            return

        logger.info("Reading candidates from %s...", jsonl_path)
        self.candidates_raw = load_jsonl(jsonl_path)
        logger.info("%d candidates loaded.", len(self.candidates_raw))

        # Build text documents
        documents = []
        ids = []
        saved_docs = {}

        for cand in self.candidates_raw:
            cid = cand.get("candidate_id")
            if not cid:
                continue
            doc = build_candidate_document(cand)
            documents.append(doc)
            ids.append(cid)
            saved_docs[cid] = doc

        self.candidate_ids = ids

        # Persist documents for inspection
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_json(saved_docs, CANDIDATE_DOCS_PATH)
        logger.info("Saved candidate documents -> %s", CANDIDATE_DOCS_PATH)

        # Encode
        logger.info("Encoding candidates (this may take a minute)...")
        self.candidate_embeddings = self.model.encode(
            documents,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        logger.info("Embeddings shape: %s", self.candidate_embeddings.shape)

        # Cache to disk
        np.save(EMBEDDINGS_CACHE_PATH, self.candidate_embeddings)
        with open(IDS_CACHE_PATH, "w") as f:
            json.dump(self.candidate_ids, f)
        logger.info("Embeddings cached to disk.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        job_description: str,
        top_k: int = 500,
    ) -> List[Dict]:
        """
        Encode the job description, compute cosine similarities,
        and return top_k candidates sorted by semantic score.

        Args:
            job_description: Raw job description text.
            top_k: Number of top candidates to return.

        Returns:
            List of dicts: {candidate_id, semantic_score, candidate_data}.

        Raises:
            AssertionError: if called before load_and_encode_candidates().
        """
        assert self.candidate_embeddings is not None, (
            "Call load_and_encode_candidates() first."
        )

        logger.info("Encoding job description...")
        jd_embedding = self.model.encode(
            [job_description], convert_to_numpy=True
        )  # shape (1, 384)

        # Cosine similarity: (1, N) → flatten to (N,)
        scores = cosine_similarity(jd_embedding, self.candidate_embeddings)[0]

        # Build index → candidate lookup
        cand_lookup = {c["candidate_id"]: c for c in self.candidates_raw}

        # Sort descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            cid = self.candidate_ids[idx]
            results.append(
                {
                    "candidate_id": cid,
                    "semantic_score": float(round(scores[idx] * 100, 4)),
                    "candidate_data": cand_lookup.get(cid, {}),
                }
            )

        logger.info(
            "Top-%d candidates retrieved. Highest semantic score: %.2f",
            top_k,
            results[0]["semantic_score"],
        )
        return results