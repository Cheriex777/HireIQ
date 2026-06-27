"""
submission_generator.py
-------------------------
Writes the final submission.csv: candidate_id,rank,score,reasoning

Uses csv.DictWriter (proper quoting) rather than manual string joins --
reasoning text contains commas, so naive `",".join(...)` output would
silently corrupt the file and likely fail the grader's CSV parser.
"""

import csv

import config


def validate_results(ranked_results, expected_n=50):
    """Fail loudly before writing a broken file, not after submission."""
    errors = []

    if len(ranked_results) != expected_n:
        errors.append(f"Expected {expected_n} rows, got {len(ranked_results)}")

    ids = [r["candidate_id"] for r in ranked_results]
    if len(ids) != len(set(ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        errors.append(f"Duplicate candidate_ids found: {dupes}")

    ranks = [r["rank"] for r in ranked_results]
    if ranks != list(range(1, len(ranked_results) + 1)):
        errors.append("Ranks are not a clean 1..N sequence")

    scores = [r["final_score"] for r in ranked_results]
    if scores != sorted(scores, reverse=True):
        errors.append("Scores are not sorted descending -- rank/score mismatch")

    for r in ranked_results:
        if not r.get("reasoning", "").strip():
            errors.append(f"Empty reasoning for {r['candidate_id']}")
            break  # one example is enough, don't flood the error list

    if errors:
        raise ValueError("Submission validation failed:\n  - " + "\n  - ".join(errors))


def write_submission(ranked_results, output_path=None, expected_n=50):
    output_path = output_path or config.SUBMISSION_PATH
    validate_results(ranked_results, expected_n=expected_n)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for r in ranked_results:
            writer.writerow({
                "candidate_id": r["candidate_id"],
                "rank": r["rank"],
                "score": r["final_score"],
                "reasoning": r["reasoning"],
            })

    print(f"Wrote validated submission -> {output_path} ({len(ranked_results)} rows)")
    return output_path
