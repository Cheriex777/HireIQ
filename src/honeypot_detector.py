"""
honeypot_detector.py
HireIQ - Honeypot / Suspicious Profile Detection

Flags candidates with unrealistic, inconsistent, or fabricated-looking
profiles. Produces a 0-1 honeypot_score and penalty to be applied
multiplicatively in final_ranker.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hireiq.honeypot_detector")
logging.basicConfig(level=logging.INFO)


# ----------------------------------------------------------------------
# Thresholds / config
# ----------------------------------------------------------------------

MAX_REALISTIC_SKILLS = 25          # beyond this, skill-padding suspected
MAX_REALISTIC_EXPERIENCE_YEARS = 30
MIN_AGE_FOR_EXPERIENCE = 18        # used if DOB/age available; else skip
SUSPICIOUS_SCORE_THRESHOLD = 0.5   # is_suspicious flag cutoff

# Titles implying high seniority that should correlate with some activity
SENIOR_TITLE_KEYWORDS = ["senior", "staff", "lead", "principal", "head", "director"]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------------
# Individual checks (each returns a 0-1 suspicion contribution)
# ----------------------------------------------------------------------

def _check_unrealistic_experience(years_of_experience: Any) -> float:
    """Flags absurd experience values (negative or implausibly high)."""
    try:
        years = _safe_float(years_of_experience, default=0.0)
        if years < 0:
            return 1.0
        if years > MAX_REALISTIC_EXPERIENCE_YEARS:
            return 0.8
        return 0.0
    except Exception as exc:
        logger.warning("Experience check failed: %s", exc)
        return 0.0


def _check_excessive_skills(skills: List[Any]) -> float:
    """Flags candidates listing an implausibly large number of skills."""
    try:
        count = len(skills) if skills else 0
        if count > MAX_REALISTIC_SKILLS:
            # scale suspicion gradually past the threshold
            overflow = count - MAX_REALISTIC_SKILLS
            return min(0.3 + 0.05 * overflow, 1.0)
        return 0.0
    except Exception as exc:
        logger.warning("Skill count check failed: %s", exc)
        return 0.0


def _check_incomplete_profile(profile: Dict[str, Any],
                               career_history: Optional[List[Any]],
                               education: Optional[List[Any]],
                               skills: Optional[List[Any]]) -> float:
    """Flags profiles missing critical fields entirely."""
    try:
        critical_missing = 0
        if not profile.get("headline"):
            critical_missing += 1
        if not profile.get("summary"):
            critical_missing += 1
        if not skills:
            critical_missing += 1
        if not career_history:
            critical_missing += 1

        if critical_missing >= 3:
            return 0.9
        elif critical_missing == 2:
            return 0.5
        elif critical_missing == 1:
            return 0.2
        return 0.0
    except Exception as exc:
        logger.warning("Profile completeness check failed: %s", exc)
        return 0.0


def _check_title_activity_mismatch(profile: Dict[str, Any],
                                    redrob_signals: Dict[str, Any]) -> float:
    """
    Flags mismatches like a senior title with near-zero engagement/activity,
    which can indicate a fabricated or dormant profile.
    """
    try:
        title = (profile.get("current_title") or "").lower()
        is_senior_title = any(kw in title for kw in SENIOR_TITLE_KEYWORDS)
        if not is_senior_title:
            return 0.0

        github_activity = _safe_float(redrob_signals.get("github_activity_score"), 0.0)
        connections = _safe_float(redrob_signals.get("connection_count"), 0.0)
        endorsements = _safe_float(redrob_signals.get("endorsements_received"), 0.0)

        # A genuinely senior person with near-zero footprint across all signals is suspicious
        if github_activity < 2 and connections < 10 and endorsements < 3:
            return 0.7
        return 0.0
    except Exception as exc:
        logger.warning("Title/activity mismatch check failed: %s", exc)
        return 0.0


def _check_abnormal_behavior(redrob_signals: Dict[str, Any]) -> float:
    """
    Flags behavioral signal combinations that look bot-like or fabricated,
    e.g. perfect 100% rates across everything, or extreme search appearance
    with zero saves/interviews.
    """
    try:
        response_rate = _safe_float(redrob_signals.get("recruiter_response_rate"), 0.0)
        interview_rate = _safe_float(redrob_signals.get("interview_completion_rate"), 0.0)
        offer_rate = _safe_float(redrob_signals.get("offer_acceptance_rate"), 0.0)
        search_appearance = _safe_float(redrob_signals.get("search_appearance_30d"), 0.0)
        saved_by_recruiters = _safe_float(redrob_signals.get("saved_by_recruiters_30d"), 0.0)

        suspicion = 0.0

        # Suspiciously perfect across the board
        if response_rate >= 0.99 and interview_rate >= 0.99 and offer_rate >= 0.99:
            suspicion = max(suspicion, 0.8)

        # High visibility but zero recruiter interest — possible scraped/dead profile
        if search_appearance > 200 and saved_by_recruiters == 0:
            suspicion = max(suspicion, 0.4)

        return suspicion
    except Exception as exc:
        logger.warning("Behavior anomaly check failed: %s", exc)
        return 0.0


def _check_inconsistent_career_history(career_history: List[Any],
                                        years_of_experience: Any) -> float:
    """
    Flags mismatches between declared years_of_experience and the
    actual span/count implied by career_history entries.
    """
    try:
        if not career_history:
            return 0.0

        declared_years = _safe_float(years_of_experience, default=0.0)
        num_roles = len(career_history)

        # Too many roles for declared experience (e.g. 6 jobs in 2 years)
        if declared_years > 0 and num_roles > 0:
            avg_tenure = declared_years / num_roles
            if avg_tenure < 0.4:  # less than ~5 months per role on average
                return 0.6

        # Career history claims more total years than declared experience
        total_years_from_history = 0.0
        for role in career_history:
            if isinstance(role, dict):
                duration = role.get("duration_years") or role.get("years")
                if duration:
                    total_years_from_history += _safe_float(duration, 0.0)

        if total_years_from_history > 0 and declared_years > 0:
            if total_years_from_history > declared_years * 1.5:
                return 0.5

        return 0.0
    except Exception as exc:
        logger.warning("Career history consistency check failed: %s", exc)
        return 0.0


# ----------------------------------------------------------------------
# Aggregator
# ----------------------------------------------------------------------

def detect_honeypot(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run all honeypot checks on a single candidate and aggregate into
    a single honeypot_score (0-1) and penalty.

    Args:
        candidate: raw candidate record.

    Returns:
        dict: {candidate_id, honeypot_score, honeypot_penalty, is_suspicious}
    """
    candidate_id = candidate.get("candidate_id", "UNKNOWN")

    try:
        profile = candidate.get("profile", {}) or {}
        career_history = candidate.get("career_history", []) or []
        education = candidate.get("education", []) or []
        redrob_signals = candidate.get("redrob_signals", {}) or {}

        raw_skills = candidate.get("skills", []) or []
        skill_names = [
            s.get("name") for s in raw_skills
            if isinstance(s, dict) and s.get("name")
        ] or raw_skills

        years_of_experience = profile.get("years_of_experience")

        checks = [
            _check_unrealistic_experience(years_of_experience),
            _check_excessive_skills(skill_names),
            _check_incomplete_profile(profile, career_history, education, skill_names),
            _check_title_activity_mismatch(profile, redrob_signals),
            _check_abnormal_behavior(redrob_signals),
            _check_inconsistent_career_history(career_history, years_of_experience),
        ]

        # Weighted average — no single check should solely tank a score,
        # but multiple simultaneous flags compound suspicion.
        honeypot_score = round(min(sum(checks) / len(checks) * 1.3, 1.0), 4)

        # Penalty is scaled slightly below raw score so it demotes rather
        # than annihilates — avoids false-positive disasters in a live demo.
        honeypot_penalty = round(min(honeypot_score * 0.8, 0.9), 4)

        is_suspicious = honeypot_score >= SUSPICIOUS_SCORE_THRESHOLD

        return {
            "candidate_id": candidate_id,
            "honeypot_score": honeypot_score,
            "honeypot_penalty": honeypot_penalty,
            "is_suspicious": is_suspicious,
        }

    except Exception as exc:
        logger.error("detect_honeypot failed for %s: %s", candidate_id, exc)
        return {
            "candidate_id": candidate_id,
            "honeypot_score": 0.0,
            "honeypot_penalty": 0.0,
            "is_suspicious": False,
        }


def detect_honeypots_batch(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run detect_honeypot over a list of candidates.

    Args:
        candidates: list of raw candidate records.

    Returns:
        list of honeypot result dicts.
    """
    results: List[Dict[str, Any]] = []
    suspicious_count = 0

    for candidate in candidates:
        cid = candidate.get("candidate_id", "UNKNOWN")
        try:
            result = detect_honeypot(candidate)
            results.append(result)
            if result["is_suspicious"]:
                suspicious_count += 1
        except Exception as exc:
            logger.error("Skipping honeypot check for %s: %s", cid, exc)
            continue

    logger.info(
        "Honeypot detection complete: %d/%d candidates flagged suspicious.",
        suspicious_count, len(results),
    )
    return results