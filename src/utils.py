"""
utils.py - Shared utilities for HireIQ pipeline.
"""

import json
import re
from typing import List, Dict, Any


def load_jsonl(filepath: str) -> List[Dict]:
    """Load a .json array or .jsonl file into a list of dicts."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("["):
        return json.loads(content)

    records = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records

def save_json(data: Any, filepath: str) -> None:
    """Save data as pretty JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to [0, 100] range."""
    if max_val == min_val:
        return 50.0
    return max(0.0, min(100.0, (value - min_val) / (max_val - min_val) * 100))


def extract_skills_from_candidate(candidate: Dict) -> List[str]:
    """Extract skill names from a candidate profile."""
    skills = candidate.get("skills", [])
    return [s["name"].lower().strip() for s in skills if s.get("name")]

def build_candidate_document(candidate: dict) -> str:
    """
    Build a rich text document for embedding from a candidate record.
    Combines headline, summary, current title, skills, career history,
    and education into a single semantically-dense string.

    Args:
        candidate: raw candidate record with profile, skills, career_history,
                   education fields.

    Returns:
        str: concatenated text document for SentenceTransformer encoding.
    """
    try:
        profile = candidate.get("profile", {}) or {}
        parts = []

        # Core profile fields
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        current_title = profile.get("current_title", "")
        current_company = profile.get("current_company", "")

        if headline:
            parts.append(headline)
        if current_title:
            parts.append(f"Current role: {current_title}" +
                         (f" at {current_company}" if current_company else ""))
        if summary:
            parts.append(summary)

        # Skills
        raw_skills = candidate.get("skills", []) or []
        skill_names = [
            s.get("name") for s in raw_skills
            if isinstance(s, dict) and s.get("name")
        ] or [s for s in raw_skills if isinstance(s, str)]
        if skill_names:
            parts.append("Skills: " + ", ".join(skill_names))

        # Career history — titles + companies + descriptions
        career_history = candidate.get("career_history", []) or []
        for role in career_history:
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

        # Education — degree + field + institution
        education = candidate.get("education", []) or []
        for edu in education:
            if not isinstance(edu, dict):
                continue
            degree = edu.get("degree", "")
            field = edu.get("field", "") or edu.get("major", "")
            institution = edu.get("institution", "") or edu.get("school", "")
            edu_line_parts = [p for p in [degree, field, institution] if p]
            if edu_line_parts:
                parts.append(", ".join(edu_line_parts))

        document = ". ".join(p.strip() for p in parts if p and p.strip())
        return document if document else "No profile information available."

    except Exception as exc:
        import logging
        logging.getLogger("hireiq.utils").error(
            "build_candidate_document failed for %s: %s",
            candidate.get("candidate_id", "UNKNOWN"), exc
        )
        return candidate.get("profile", {}).get("headline", "") or "No profile information available."

def get_years_of_experience(candidate: Dict) -> float:
    """Safely extract years_of_experience from candidate."""
    return candidate.get("profile", {}).get("years_of_experience", 0) or 0