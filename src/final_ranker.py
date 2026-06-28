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

from src.config import MAX_PENALTY_CAP, RANKING_WEIGHTS, REQUIRED_FEATURE_KEYS

logger = logging.getLogger("hireiq.final_ranker")
logging.basicConfig(level=logging.INFO)


class HybridRanker:
    """Combines multi-signal features and a honeypot penalty into a final score.

    Attributes:
        weights: Mapping of feature-score key to its ranking weight. Defaults
            to ``config.RANKING_WEIGHTS`` (the production weighting scheme).
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        """Initialize the ranker.

        Args:
            weights: Optional override for the ranking weights. If omitted,
                the project-wide defaults from ``config.RANKING_WEIGHTS``
                are used.
        """
        self.weights: Dict[str, float] = weights or RANKING_WEIGHTS
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Warn if the configured weights don't sum to ~1.0.

        Does not raise — ranking proceeds with whatever weights are given,
        but a loud warning is logged so misconfiguration is easy to spot.
        """
        try:
            total = sum(self.weights.values())
            if not (0.99 <= total <= 1.01):
                logger.warning(
                    "Ranking weights sum to %.4f, expected ~1.0. Proceeding anyway.",
                    total,
                )
        except TypeError as exc:
            logger.error("Weight validation failed: %s", exc)

    def _validate_feature_row(self, row: Dict[str, Any]) -> bool:
        """Check that a candidate row has all required feature keys.

        Args:
            row: A single candidate's feature dict.

        Returns:
            True if all required keys are present, False otherwise.
        """
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
        """Weighted combination of all 6 signals, before honeypot penalty.

        Args:
            features: Candidate feature dict containing semantic_score,
                skill_score, experience_score, behavior_score, title_score,
                and profile_score.

        Returns:
            Base score clamped to [0.0, 100.0], rounded to 2 decimals.
        """
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
        """Apply honeypot penalty multiplicatively to the base score.

        Args:
            features: Candidate feature dict, must include ``honeypot_penalty``.

        Returns:
            Final score clamped to [0.0, 100.0], rounded to 2 decimals.
        """
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
        """Score and rank all candidates, returning the top_n with rank assigned.

        Args:
            feature_rows: List of dicts containing feature scores AND
                honeypot_score/honeypot_penalty/is_suspicious (merged in
                beforehand by run.py).
            top_n: Number of top candidates to return.

        Returns:
            List of dicts: ``{..., base_score, final_score, rank}``.
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
    """Convenience function wrapper around HybridRanker for direct use in run.py.

    Args:
        feature_rows: List of candidate feature dicts to rank.
        top_n: Number of top candidates to return.
        weights: Optional override for ranking weights.

    Returns:
        List of ranked candidate dicts with base_score, final_score, and rank.
    """
    ranker = HybridRanker(weights=weights)
    return ranker.rank(feature_rows, top_n=top_n)