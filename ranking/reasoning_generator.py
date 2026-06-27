"""
reasoning_generator.py
------------------------
Generates the human-readable `reasoning` column for each ranked candidate,
in the recruiter-voice style the spec asks for, e.g.:

  "Candidate has 7.8 years of experience in search and retrieval systems,
   strong expertise in FAISS, Information Retrieval and Semantic Search,
   and demonstrates strong recruiter engagement metrics."

Deliberately template-based, not an LLM call: it only needs to run on the
top 100 results of a single ranking, so cost isn't the concern -- the
concern is reproducibility. A submission whose explanations come from an
LLM call means re-running your own pipeline can produce different text,
and grading it depends on an external API being reachable. Template-based
reasoning is deterministic, fast, and auditable -- every sentence traces
back to the exact score component that produced it.

This intentionally does NOT only say nice things: a candidate can land in
the top 100 on the strength of 4 components while being weak on the 5th
(e.g. under the years requirement, or thin skill coverage), and the
reasoning says so. A recruiter reading 100 of these should be able to spot
the borderline ones, not just see uniform praise.

If you want richer prose later, this is the one place to swap in an LLM
call per top-100 candidate -- the function signature wouldn't need to change.
"""

import config

_DOMAIN_KEYWORDS = [
    (("search", "retrieval", "information retrieval"), "search and retrieval systems"),
    (("recommendation", "recsys"), "recommendation systems"),
    (("nlp", "natural language"), "natural language processing"),
    (("ranking",), "ranking and relevance systems"),
    (("research",), "applied AI research"),
    (("data scien",), "data science and modeling"),
]


def _domain_phrase(record):
    title = (record.get("current_title") or "").lower()
    headline = (record.get("headline") or "").lower()
    haystack = title + " " + headline

    for keywords, phrase in _DOMAIN_KEYWORDS:
        if any(k in haystack for k in keywords):
            return phrase

    if title:
        return f"{title} work"
    return "machine learning engineering"


def _skill_list_phrase(matched_skills, max_skills=3):
    clean = sorted({s.replace(" (mentioned)", "") for s in matched_skills})
    if not clean:
        return None
    shown = clean[:max_skills]
    if len(shown) == 1:
        return shown[0].title()
    if len(shown) == 2:
        return f"{shown[0].title()} and {shown[1].title()}"
    return ", ".join(s.title() for s in shown[:-1]) + f", and {shown[-1].title()}"


def _behavioral_phrase(record):
    signals = record.get("redrob_signals", {})
    response = signals.get("recruiter_response_rate", 0) or 0
    interview = signals.get("interview_completion_rate", 0) or 0

    if response >= 0.7 and interview >= 0.7:
        return "strong recruiter engagement metrics"
    if response >= 0.7:
        return "a strong recruiter response rate"
    if interview >= 0.7:
        return "a high interview completion rate"
    if signals.get("open_to_work_flag"):
        return "active openness to new opportunities"
    return "moderate recruiter engagement metrics"


def _main_sentence(record, matched_skills):
    years = record.get("years_of_experience", 0) or 0
    domain = _domain_phrase(record)
    skills_phrase = _skill_list_phrase(matched_skills)
    behavioral = _behavioral_phrase(record)

    if skills_phrase:
        return (
            f"Candidate has {years} years of experience in {domain}, "
            f"strong expertise in {skills_phrase}, and demonstrates {behavioral}."
        )
    return (
        f"Candidate has {years} years of experience in {domain}, "
        f"and demonstrates {behavioral}, though direct overlap with the "
        f"JD's required skill list is limited."
    )


def _caveat_sentence(record, comp, required_years):
    """Honest gaps, only stated when they're actually true -- not a
    boilerplate disclaimer on every row."""
    notes = []

    years = record.get("years_of_experience", 0) or 0
    if years < required_years:
        notes.append(f"experience ({years} years) is below the {required_years}+ year requirement")

    if comp["semantic"] < 0.25:
        notes.append("overall semantic alignment with the job description is comparatively weak")

    if comp["skill"] < 0.35:
        notes.append("formal skill-list coverage of the JD's required skills is thin")

    if not notes:
        return None
    if len(notes) == 1:
        return f"Worth noting: {notes[0]}."
    return "Worth noting: " + "; ".join(notes) + "."


def generate_reasoning(ranked_result, required_years=None):
    """ranked_result is one element from HybridRanker.rank()'s output list."""
    required_years = required_years or config.REQUIRED_YEARS
    record = ranked_result["candidate"]
    comp = ranked_result["component_scores"]

    main = _main_sentence(record, ranked_result["matched_skills"])
    caveat = _caveat_sentence(record, comp, required_years)

    return f"{main} {caveat}" if caveat else main


def attach_reasoning(ranked_results, required_years=None):
    """Mutates each result dict in place, adding a 'reasoning' key.
    Returns the same list for convenience."""
    for r in ranked_results:
        r["reasoning"] = generate_reasoning(r, required_years=required_years)
    return ranked_results
