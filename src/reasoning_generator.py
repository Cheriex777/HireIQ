"""
reasoning_generator.py
HireIQ - Explainability Engine

Generates a concise, human-readable one-line reasoning for why each
candidate was ranked where they were. Includes an honest caveat clause
when a candidate is genuinely weak on a specific signal, instead of
only ever praising -- a recruiter reading 100 of these should be able
to spot the borderline candidates, not see uniform positivity.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hireiq.reasoning_generator")
logging.basicConfig(level=logging.INFO)


# Thresholds used to decide which signals are "strong enough" to mention
SEMANTIC_HIGH = 80.0
SKILL_HIGH = 70.0
BEHAVIOR_HIGH = 70.0
TITLE_HIGH = 80.0
EXPERIENCE_HIGH = 80.0

# Thresholds below which a signal is weak enough to mention as a caveat
SEMANTIC_LOW = 40.0
SKILL_LOW = 35.0
EXPERIENCE_LOW_YEARS = 5  # JD's stated minimum

MAX_REASON_CLAUSES = 4


def _safe_get(features: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(features.get(key, default))
    except (TypeError, ValueError):
        return default


def _experience_clause(candidate: Dict[str, Any], experience_score: float) -> Optional[str]:
    try:
        years = candidate.get("profile", {}).get("years_of_experience")
        if years is None:
            return None
        years = float(years)
        if experience_score >= EXPERIENCE_HIGH:
            return f"{int(years)} years of strong relevant experience"
        if experience_score >= 50:
            return f"{int(years)} years of relevant experience"
        return None
    except Exception:
        return None


def _skill_clause(candidate: Dict[str, Any], jd_skills: List[str], skill_score: float) -> Optional[str]:
    try:
        candidate_skills = candidate.get("skills", []) or []
        cand_set = {s.strip().lower() for s in candidate_skills if isinstance(s, str)}
        jd_set = {s.strip().lower() for s in jd_skills if isinstance(s, str)}
        overlap = sorted(jd_set & cand_set)

        if not overlap:
            return None

        shown = overlap[:3]
        skill_str = ", ".join(s.upper() if len(s) <= 5 else s.title() for s in shown)

        if skill_score >= SKILL_HIGH:
            return f"strong expertise in {skill_str}"
        if skill_score >= 40:
            return f"working knowledge of {skill_str}"
        return None
    except Exception as exc:
        logger.warning("Skill clause generation failed: %s", exc)
        return None


def _title_clause(candidate: Dict[str, Any], title_score: float) -> Optional[str]:
    try:
        title = candidate.get("profile", {}).get("current_title")
        if not title:
            return None
        if title_score >= TITLE_HIGH:
            return f"currently a {title}"
        return None
    except Exception:
        return None


def _behavior_clause(behavior_score: float) -> Optional[str]:
    if behavior_score >= BEHAVIOR_HIGH:
        return "high recruiter engagement"
    if behavior_score >= 40:
        return "moderate recruiter engagement"
    return None


def _semantic_clause(semantic_score: float) -> Optional[str]:
    if semantic_score >= SEMANTIC_HIGH:
        return "excellent semantic match to job description"
    if semantic_score >= 50:
        return "good semantic relevance to job description"
    return None


def _caveat_clause(
    candidate: Dict[str, Any],
    semantic_score: float,
    skill_score: float,
) -> Optional[str]:
    """
    Honest gap-flagging -- only stated when actually true, not a
    boilerplate disclaimer on every row. Mirrors a single weak signal
    so a recruiter scanning the list can spot borderline candidates.
    """
    try:
        years = candidate.get("profile", {}).get("years_of_experience")
        notes = []

        if years is not None:
            years = float(years)
            if years < EXPERIENCE_LOW_YEARS:
                notes.append(f"experience ({int(years)} years) is below the {EXPERIENCE_LOW_YEARS}+ year requirement")

        if semantic_score < SEMANTIC_LOW:
            notes.append("overall semantic alignment with the job description is comparatively weak")

        if skill_score < SKILL_LOW:
            notes.append("formal skill-list coverage of the JD's required skills is thin")

        if not notes:
            return None

        # Only ever surface the single most relevant gap -- avoid piling on
        return f"worth noting: {notes[0]}"

    except Exception as exc:
        logger.warning("Caveat clause generation failed: %s", exc)
        return None


def generate_reasoning(
    candidate: Dict[str, Any],
    features: Dict[str, Any],
    jd_skills: List[str],
    include_caveats: bool = True,
) -> str:
    """
    Build a one-line, human-readable explanation for a candidate's rank.

    Args:
        candidate: raw candidate record (profile, skills, etc.)
        features: feature dict from feature_extractor (scores).
        jd_skills: skills parsed from the job description.
        include_caveats: if True, appends an honest gap-flag sentence
                          when a candidate is genuinely weak on a signal.
                          Set False to restore the original praise-only behavior.

    Returns:
        Single-line reasoning string. Falls back to a generic message on error.
    """
    candidate_id = candidate.get("candidate_id", features.get("candidate_id", "UNKNOWN"))

    try:
        semantic_score = _safe_get(features, "semantic_score")
        skill_score = _safe_get(features, "skill_score")
        experience_score = _safe_get(features, "experience_score")
        behavior_score = _safe_get(features, "behavior_score")
        title_score = _safe_get(features, "title_score")

        clause_builders = [
            _skill_clause(candidate, jd_skills, skill_score),
            _experience_clause(candidate, experience_score),
            _semantic_clause(semantic_score),
            _behavior_clause(behavior_score),
            _title_clause(candidate, title_score),
        ]

        clauses = [c for c in clause_builders if c]

        if not clauses:
            main = "Moderate overall fit based on combined profile signals."
        else:
            clauses = clauses[:MAX_REASON_CLAUSES]
            main = ", ".join(clauses)
            main = main[0].upper() + main[1:] + "."

        if include_caveats:
            caveat = _caveat_clause(candidate, semantic_score, skill_score)
            if caveat:
                return f"{main} {caveat[0].upper()}{caveat[1:]}."

        return main

    except Exception as exc:
        logger.error("generate_reasoning failed for candidate %s: %s", candidate_id, exc)
        return "Ranked based on combined semantic, skill, and behavioral signals."


def generate_reasoning_batch(
    candidates: List[Dict[str, Any]],
    ranked_features: List[Dict[str, Any]],
    jd_skills: List[str],
    include_caveats: bool = True,
) -> Dict[str, str]:
    """
    Generate reasoning strings for a batch of ranked candidates.

    Args:
        candidates: list of raw candidate records.
        ranked_features: list of feature/score dicts (from final_ranker output).
        jd_skills: skills parsed from job description.
        include_caveats: passed through to generate_reasoning() for every candidate.

    Returns:
        dict mapping candidate_id -> reasoning string.
    """
    candidate_lookup: Dict[str, Dict[str, Any]] = {
        c.get("candidate_id", "UNKNOWN"): c for c in candidates
    }

    reasoning_map: Dict[str, str] = {}

    for features in ranked_features:
        cid = features.get("candidate_id", "UNKNOWN")
        candidate = candidate_lookup.get(cid, {"candidate_id": cid})
        try:
            reasoning_map[cid] = generate_reasoning(
                candidate, features, jd_skills, include_caveats=include_caveats
            )
        except Exception as exc:
            logger.error("Failed to generate reasoning for %s: %s", cid, exc)
            reasoning_map[cid] = "Ranked based on combined profile signals."

    logger.info("Generated reasoning for %d candidates.", len(reasoning_map))
    return reasoning_map