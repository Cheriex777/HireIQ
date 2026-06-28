"""
submission_generator.py
HireIQ - Submission Generator

Builds the final submission.csv from ranked candidates and their
generated reasoning strings. Validates the result BEFORE writing,
so a broken pipeline fails loudly instead of silently shipping a
corrupted CSV.
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hireiq.submission_generator")
logging.basicConfig(level=logging.INFO)


SUBMISSION_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]

DEFAULT_OUTPUT_PATH = os.path.join("outputs", "submission.csv")


def _validate_row(row: Dict[str, Any]) -> bool:
    required = ("candidate_id", "rank", "final_score")
    missing = [k for k in required if k not in row]
    if missing:
        logger.warning(
            "Candidate %s missing keys %s; skipping in submission.",
            row.get("candidate_id", "UNKNOWN"),
            missing,
        )
        return False
    return True


def validate_submission_rows(rows: List[Dict[str, Any]], expected_n: Optional[int] = None) -> None:
    """
    Fail loudly BEFORE writing a broken file, not after submission.

    Checks:
      - row count matches expected_n (if given)
      - no duplicate candidate_ids
      - ranks form a clean 1..N sequence
      - scores are sorted descending (rank/score consistency)
      - no empty reasoning text

    Raises:
        ValueError: with a combined list of every issue found.
    """
    errors: List[str] = []

    if expected_n is not None and len(rows) != expected_n:
        errors.append(f"Expected {expected_n} rows, got {len(rows)}")

    ids = [r["candidate_id"] for r in rows]
    if len(ids) != len(set(ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        errors.append(f"Duplicate candidate_ids found: {dupes}")

    ranks = [r["rank"] for r in rows]
    if ranks != list(range(1, len(rows) + 1)):
        errors.append("Ranks are not a clean 1..N sequence")

    scores = [r["score"] for r in rows]
    if scores != sorted(scores, reverse=True):
        errors.append("Scores are not sorted descending -- rank/score mismatch")

    for r in rows:
        if not str(r.get("reasoning", "")).strip():
            errors.append(f"Empty reasoning found for candidate {r.get('candidate_id', 'UNKNOWN')}")
            break  # one example is enough, don't flood the error list

    if errors:
        raise ValueError("Submission validation failed:\n  - " + "\n  - ".join(errors))

    logger.info("Submission validation passed: %d rows, all checks OK.", len(rows))


def build_submission_rows(
    ranked_candidates: List[Dict[str, Any]],
    reasoning_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Merge ranked candidate scores with reasoning text into final submission rows.

    Args:
        ranked_candidates: list of dicts from final_ranker.rank() output
                            (must contain candidate_id, rank, final_score).
        reasoning_map: dict mapping candidate_id -> reasoning string.

    Returns:
        list of dicts with keys: candidate_id, rank, score, reasoning.
    """
    rows: List[Dict[str, Any]] = []

    for cand in ranked_candidates:
        if not _validate_row(cand):
            continue

        cid = cand.get("candidate_id", "UNKNOWN")
        try:
            rows.append({
                "candidate_id": cid,
                "rank": int(cand.get("rank", 0)),
                "score": round(float(cand.get("final_score", 0.0)), 2),
                "reasoning": reasoning_map.get(cid, "Ranked based on combined profile signals."),
            })
        except Exception as exc:
            logger.error("Failed to build submission row for %s: %s", cid, exc)
            continue

    rows.sort(key=lambda r: r["rank"])
    logger.info("Built %d submission rows.", len(rows))
    return rows


def write_submission_csv(
    rows: List[Dict[str, Any]],
    output_path: str = DEFAULT_OUTPUT_PATH,
    expected_n: Optional[int] = None,
    skip_validation: bool = False,
) -> str:
    """
    Validate (unless skipped), then write submission rows to a CSV file.

    Args:
        rows: list of dicts with keys candidate_id, rank, score, reasoning.
        output_path: destination path for submission.csv.
        expected_n: if given, validation fails if row count doesn't match
                     (e.g. pass 100 to enforce exactly top-100).
        skip_validation: set True only for quick debug runs on small samples
                          where a mismatched row count is expected.

    Returns:
        The output_path written to.

    Raises:
        ValueError: if validation fails (row count, duplicates, rank/score
                     order, or empty reasoning).
        IOError: if the file cannot be written.
    """
    if not skip_validation:
        validate_submission_rows(rows, expected_n=expected_n)

    try:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({col: row.get(col, "") for col in SUBMISSION_COLUMNS})

        logger.info("Submission written to %s (%d rows).", output_path, len(rows))
        return output_path

    except Exception as exc:
        logger.error("Failed to write submission CSV to %s: %s", output_path, exc)
        raise IOError(f"Could not write submission CSV: {exc}") from exc


def generate_submission(
    ranked_candidates: List[Dict[str, Any]],
    reasoning_map: Dict[str, str],
    output_path: str = DEFAULT_OUTPUT_PATH,
    expected_n: Optional[int] = None,
) -> str:
    """
    End-to-end: build rows from ranked candidates + reasoning, validate,
    write to CSV.

    Args:
        ranked_candidates: output from final_ranker.rank().
        reasoning_map: output from reasoning_generator.generate_reasoning_batch().
        output_path: path to write submission.csv.
        expected_n: pass e.g. 100 to enforce exactly that many rows; leave
                     None to skip the row-count check (still validates
                     duplicates, rank order, score order, empty reasoning).

    Returns:
        Path to the written CSV file.
    """
    rows = build_submission_rows(ranked_candidates, reasoning_map)
    if not rows:
        logger.warning("No valid rows to write; submission.csv will be empty (header only).")
        return write_submission_csv(rows, output_path=output_path, skip_validation=True)

    return write_submission_csv(rows, output_path=output_path, expected_n=expected_n)