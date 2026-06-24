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


def build_candidate_document(candidate: Dict) -> str:
    """
    Combine headline + summary + skills + titles into a single searchable text.
    Used for generating embeddings.
    """
    profile = candidate.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    current_title = profile.get("current_title", "")

    skills = extract_skills_from_candidate(candidate)
    skills_text = " ".join(skills)

    # Include career history descriptions for richer context
    career_snippets = []
    for job in candidate.get("career_history", []):
        title = job.get("title", "")
        desc = job.get("description", "")
        if title:
            career_snippets.append(title)
        if desc:
            career_snippets.append(desc[:200])  # first 200 chars

    career_text = " ".join(career_snippets)

    document = f"{current_title} {headline} {summary} {skills_text} {career_text}"
    # Clean extra whitespace
    document = re.sub(r"\s+", " ", document).strip()
    return document


def get_years_of_experience(candidate: Dict) -> float:
    """Safely extract years_of_experience from candidate."""
    return candidate.get("profile", {}).get("years_of_experience", 0) or 0