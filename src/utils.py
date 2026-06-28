"""
utils.py - Shared utilities for HireIQ pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger("hireiq.utils")


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSON array or JSONL file into a list of dicts.

    Supports both standard JSON arrays (``[{...}, {...}]``) and newline-
    delimited JSONL files where each line is an independent JSON object.

    Args:
        filepath: Path to the ``.json`` or ``.jsonl`` file.

    Returns:
        List of parsed record dicts.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        json.JSONDecodeError: If any record cannot be parsed.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("["):
        return json.loads(content)

    records: List[Dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def save_json(data: Any, filepath: str) -> None:
    """Serialize *data* to a pretty-printed JSON file.

    Args:
        data: Any JSON-serializable object.
        filepath: Destination file path. Parent directories must exist.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Linearly scale *value* from [min_val, max_val] to [0, 100].

    If *min_val* equals *max_val* (zero-range input), returns 50.0 to
    avoid a division-by-zero and place the value at the midpoint.

    Args:
        value: Raw value to normalize.
        min_val: Lower bound of the input range.
        max_val: Upper bound of the input range.

    Returns:
        Normalized float clamped to ``[0.0, 100.0]``.
    """
    if max_val == min_val:
        return 50.0
    return max(0.0, min(100.0, (value - min_val) / (max_val - min_val) * 100))


def extract_skills_from_candidate(candidate: Dict[str, Any]) -> List[str]:
    """Return a lowercased list of skill names from a candidate profile.

    Args:
        candidate: Raw candidate record containing an optional ``skills``
            list of dicts, each with at least a ``"name"`` key.

    Returns:
        List of normalized (lowercase, stripped) skill name strings.
    """
    skills = candidate.get("skills", [])
    return [s["name"].lower().strip() for s in skills if s.get("name")]


def build_candidate_document(candidate: Dict[str, Any]) -> str:
    """Build a semantically dense text document from a candidate record.

    Combines headline, current role, summary, skills, career history, and
    education into a single string suitable for SentenceTransformer encoding.
    Falls back to the profile headline (or a placeholder) on any error so
    that downstream embedding never receives an empty string.

    Args:
        candidate: Raw candidate record with optional ``profile``, ``skills``,
            ``career_history``, and ``education`` fields.

    Returns:
        Concatenated text document for embedding.
    """
    try:
        profile = candidate.get("profile", {}) or {}
        parts: List[str] = []

        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        current_title = profile.get("current_title", "")
        current_company = profile.get("current_company", "")

        if headline:
            parts.append(headline)
        if current_title:
            role_line = f"Current role: {current_title}"
            if current_company:
                role_line += f" at {current_company}"
            parts.append(role_line)
        if summary:
            parts.append(summary)

        # Skills
        raw_skills = candidate.get("skills", []) or []
        skill_names: List[str] = (
            [s.get("name") for s in raw_skills if isinstance(s, dict) and s.get("name")]
            or [s for s in raw_skills if isinstance(s, str)]
        )
        if skill_names:
            parts.append("Skills: " + ", ".join(skill_names))

        # Career history
        for role in candidate.get("career_history", []) or []:
            if not isinstance(role, dict):
                continue
            role_title = role.get("title", "")
            role_company = role.get("company", "")
            role_desc = role.get("description", "")
            role_line_parts = [p for p in [role_title, role_company] if p]
            if role_line_parts:
                line = " at ".join(role_line_parts)
                if role_desc:
                    line += f": {role_desc}"
                parts.append(line)

        # Education
        for edu in candidate.get("education", []) or []:
            if not isinstance(edu, dict):
                continue
            degree = edu.get("degree", "")
            field = edu.get("field", "") or edu.get("major", "")
            institution = edu.get("institution", "") or edu.get("school", "")
            edu_parts = [p for p in [degree, field, institution] if p]
            if edu_parts:
                parts.append(", ".join(edu_parts))

        document = ". ".join(p.strip() for p in parts if p and p.strip())
        return document if document else "No profile information available."

    except Exception as exc:
        logger.error(
            "build_candidate_document failed for %s: %s",
            candidate.get("candidate_id", "UNKNOWN"),
            exc,
            exc_info=True,
        )
        return (
            candidate.get("profile", {}).get("headline", "")
            or "No profile information available."
        )


def get_years_of_experience(candidate: Dict[str, Any]) -> float:
    """Safely extract years of experience from a candidate record.

    Args:
        candidate: Raw candidate record with an optional ``profile`` dict
            containing a ``years_of_experience`` numeric field.

    Returns:
        Years of experience as a float, defaulting to ``0.0`` if absent or
        ``None``.
    """
    return candidate.get("profile", {}).get("years_of_experience", 0) or 0