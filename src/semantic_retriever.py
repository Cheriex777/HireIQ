#Step 1: embeddings + cosine search


"""
semantic_retriever.py - Generates candidate embeddings and retrieves
top-N candidates semantically closest to a job description.

Usage:
    from src.semantic_retriever import SemanticRetriever

    retriever = SemanticRetriever()
    retriever.load_and_encode_candidates("data/candidates.jsonl")
    results = retriever.retrieve(job_description, top_k=500)
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import load_jsonl, build_candidate_document, save_json


MODEL_NAME = "all-MiniLM-L6-v2"
CANDIDATE_DOCS_PATH = "output/candidate_documents.json"
EMBEDDINGS_CACHE_PATH = "output/candidate_embeddings.npy"
IDS_CACHE_PATH = "output/candidate_ids.json"


class SemanticRetriever:
    """
    Encodes candidate profiles using SentenceTransformer and performs
    cosine-similarity-based retrieval against a job description.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"[SemanticRetriever] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.candidate_ids: List[str] = []
        self.candidate_embeddings: np.ndarray = None
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
        """
        if (
            not force_recompute
            and os.path.exists(EMBEDDINGS_CACHE_PATH)
            and os.path.exists(IDS_CACHE_PATH)
        ):
            print("[SemanticRetriever] Loading cached embeddings …")
            self.candidate_embeddings = np.load(EMBEDDINGS_CACHE_PATH)
            with open(IDS_CACHE_PATH, "r") as f:
                self.candidate_ids = json.load(f)
            self.candidates_raw = load_jsonl(jsonl_path)
            print(
                f"[SemanticRetriever] Loaded {len(self.candidate_ids)} candidates "
                f"with embeddings shape {self.candidate_embeddings.shape}"
            )
            return

        print(f"[SemanticRetriever] Reading candidates from {jsonl_path} …")
        self.candidates_raw = load_jsonl(jsonl_path)
        print(f"[SemanticRetriever] {len(self.candidates_raw)} candidates loaded.")

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
        os.makedirs("output", exist_ok=True)
        save_json(saved_docs, CANDIDATE_DOCS_PATH)
        print(f"[SemanticRetriever] Saved candidate documents → {CANDIDATE_DOCS_PATH}")

        # Encode
        print("[SemanticRetriever] Encoding candidates (this may take a minute) …")
        self.candidate_embeddings = self.model.encode(
            documents,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        print(f"[SemanticRetriever] Embeddings shape: {self.candidate_embeddings.shape}")

        # Cache to disk
        np.save(EMBEDDINGS_CACHE_PATH, self.candidate_embeddings)
        with open(IDS_CACHE_PATH, "w") as f:
            json.dump(self.candidate_ids, f)
        print("[SemanticRetriever] Embeddings cached to disk.")

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

        Returns list of dicts:
            {candidate_id, semantic_score, candidate_data}
        """
        assert self.candidate_embeddings is not None, (
            "Call load_and_encode_candidates() first."
        )

        print(f"[SemanticRetriever] Encoding job description …")
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

        print(
            f"[SemanticRetriever] Top-{top_k} candidates retrieved. "
            f"Highest semantic score: {results[0]['semantic_score']:.2f}"
        )
        return results