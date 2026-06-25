"""
final_ranker.py
HireIQ - Hybrid Ranking Engine

Combines semantic, skill, experience, behavior, title, and profile-
completeness scores into a base score, then applies a multiplicative
honeypot penalty to demote suspicious candidates before ranking.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hireiq.final_ranker")
logging.basicConfig(level=logging.INFO)


DEFAULT_WEIGHTS: Dict[str, float] = {
    "semantic_score": 0.35,
    "skill_score": 0.27,
    "experience_score": 0.15,
    "behavior_score": 0.10,
    "title_score": 0.08,
    "profile_score": 0.05,
}

REQUIRED_FEATURE_KEYS = (
    "candidate_id",
    "semantic_score",
    "skill_score",
    "experience_score",
    "behavior_score",
    "title_score",
)

MAX_PENALTY_CAP = 0.9  # never reduce a score by more than 90%


class HybridRanker:
    """Combines multi-signal features + honeypot penalty into a final score."""

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or DEFAULT_WEIGHTS
        self._validate_weights()

    def _validate_weights(self) -> None:
        try:
            total = sum(self.weights.values())
            if not (0.99 <= total <= 1.01):
                logger.warning(
                    "Ranking weights sum to %.4f, expected ~1.0. Proceeding anyway.",
                    total,
                )
        except Exception as exc:
            logger.error("Weight validation failed: %s", exc)

    def _validate_feature_row(self, row: Dict[str, Any]) -> bool:
        missing = [k for k in REQUIRED_FEATURE_KEYS if k not in row]
        if missing:
            logger.warning(
                "Candidate %s missing keys %s; skipping.",
                row.get("candidate_id", "UNKNOWN"),
                missing,
            )
            return False
        return True

    def calculate_base_score(self, features: Dict[str, Any]) -> float:
        """Weighted combination of all 6 signals, before honeypot penalty."""
        try:
            score = (
                self.weights["semantic_score"] * float(features.get("semantic_score", 0.0))
                + self.weights["skill_score"] * float(features.get("skill_score", 0.0))
                + self.weights["experience_score"] * float(features.get("experience_score", 0.0))
                + self.weights["behavior_score"] * float(features.get("behavior_score", 0.0))
                + self.weights["title_score"] * float(features.get("title_score", 0.0))
                + self.weights["profile_score"] * float(features.get("profile_score", 0.0))
            )
            return round(min(max(score, 0.0), 100.0), 2)
        except Exception as exc:
            logger.error(
                "calculate_base_score failed for candidate %s: %s",
                features.get("candidate_id", "UNKNOWN"),
                exc,
            )
            return 0.0

    def calculate_final_score(self, features: Dict[str, Any]) -> float:
        """Apply honeypot penalty multiplicatively to the base score."""
        try:
            base_score = self.calculate_base_score(features)
            penalty = float(features.get("honeypot_penalty", 0.0))
            penalty = min(max(penalty, 0.0), MAX_PENALTY_CAP)

            final = base_score * (1.0 - penalty)
            return round(min(max(final, 0.0), 100.0), 2)
        except Exception as exc:
            logger.error(
                "calculate_final_score failed for candidate %s: %s",
                features.get("candidate_id", "UNKNOWN"),
                exc,
            )
            return 0.0

    def rank(
        self,
        feature_rows: List[Dict[str, Any]],
        top_n: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Score and rank all candidates, returning the top_n with rank assigned.

        Args:
            feature_rows: list of dicts containing feature scores AND
                          honeypot_score/honeypot_penalty/is_suspicious
                          (merged in beforehand by run.py).
            top_n: number of top candidates to return.

        Returns:
            list of dicts: {..., base_score, final_score, rank}
        """
        scored: List[Dict[str, Any]] = []

        for row in feature_rows:
            if not self._validate_feature_row(row):
                continue
            try:
                base_score = self.calculate_base_score(row)
                final_score = self.calculate_final_score(row)

                enriched = dict(row)
                enriched["base_score"] = base_score
                enriched["final_score"] = final_score
                scored.append(enriched)
            except Exception as exc:
                logger.error(
                    "Failed to score candidate %s: %s",
                    row.get("candidate_id", "UNKNOWN"),
                    exc,
                )
                continue

        scored.sort(key=lambda x: x["final_score"], reverse=True)

        top_candidates = scored[:top_n]
        for idx, cand in enumerate(top_candidates, start=1):
            cand["rank"] = idx

        suspicious_in_top = sum(1 for c in top_candidates if c.get("is_suspicious"))
        logger.info(
            "Ranked %d candidates, returning top %d (%d flagged suspicious in top results).",
            len(scored),
            len(top_candidates),
            suspicious_in_top,
        )
        return top_candidates


def rank_candidates(
    feature_rows: List[Dict[str, Any]],
    top_n: int = 100,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Convenience function wrapper around HybridRanker for direct use in run.py."""
    ranker = HybridRanker(weights=weights)
    return ranker.rank(feature_rows, top_n=top_n)