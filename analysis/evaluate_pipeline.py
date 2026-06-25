"""
analysis/evaluate_pipeline.py
HireIQ - Pipeline Evaluation / Summary Statistics

Generates dataset-level insights for the final submission:
top titles, top skills, average experience, average behavior score,
suspicious profile count, and score distribution — for use in the
presentation and sanity-checking the pipeline output.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

logger = logging.getLogger("hireiq.evaluate_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def load_candidates(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load raw candidates from a .jsonl file."""
    candidates = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
        logger.info("Loaded %d candidates from %s", len(candidates), jsonl_path)
    except Exception as exc:
        logger.error("Failed to load candidates from %s: %s", jsonl_path, exc)
    return candidates


def load_submission(csv_path: str) -> pd.DataFrame:
    """Load the final ranked submission CSV."""
    try:
        df = pd.read_csv(csv_path)
        logger.info("Loaded submission with %d rows from %s", len(df), csv_path)
        return df
    except Exception as exc:
        logger.error("Failed to load submission CSV %s: %s", csv_path, exc)
        return pd.DataFrame()


def get_top_titles(candidates: List[Dict[str, Any]], top_n: int = 10) -> List[tuple]:
    """Most common current_title values across all candidates."""
    try:
        titles = [
            c.get("profile", {}).get("current_title", "Unknown")
            for c in candidates
            if c.get("profile", {}).get("current_title")
        ]
        return Counter(titles).most_common(top_n)
    except Exception as exc:
        logger.error("get_top_titles failed: %s", exc)
        return []


def get_top_skills(candidates: List[Dict[str, Any]], top_n: int = 15) -> List[tuple]:
    """Most common skill names across all candidates."""
    try:
        all_skills = []
        for c in candidates:
            raw_skills = c.get("skills", []) or []
            for s in raw_skills:
                name = s.get("name") if isinstance(s, dict) else s
                if name:
                    all_skills.append(name.strip())
        return Counter(all_skills).most_common(top_n)
    except Exception as exc:
        logger.error("get_top_skills failed: %s", exc)
        return []


def get_avg_experience(candidates: List[Dict[str, Any]]) -> float:
    """Average years_of_experience across all candidates."""
    try:
        years = [
            float(c.get("profile", {}).get("years_of_experience", 0))
            for c in candidates
            if c.get("profile", {}).get("years_of_experience") is not None
        ]
        return round(sum(years) / len(years), 2) if years else 0.0
    except Exception as exc:
        logger.error("get_avg_experience failed: %s", exc)
        return 0.0


def get_avg_behavior_signals(candidates: List[Dict[str, Any]]) -> Dict[str, float]:
    fields = [
        "recruiter_response_rate", "github_activity_score",
        "search_appearance_30d", "saved_by_recruiters_30d",
        "interview_completion_rate", "offer_acceptance_rate",
    ]
    averages = {}
    try:
        for field in fields:
            values = []
            for c in candidates:
                val = c.get("redrob_signals", {}).get(field)
                if val is not None:
                    val = float(val)
                    if val < 0:
                        continue  # skip sentinel/missing values, don't count as 0 or negative
                    values.append(val)
            averages[field] = round(sum(values) / len(values), 4) if values else 0.0
    except Exception as exc:
        logger.error("get_avg_behavior_signals failed: %s", exc)
    return averages

def get_suspicious_count(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run honeypot detection across all candidates and summarize results."""
    try:
        from src.honeypot_detector import detect_honeypots_batch
        results = detect_honeypots_batch(candidates)
        suspicious = [r for r in results if r.get("is_suspicious")]
        return {
            "total_candidates": len(results),
            "suspicious_count": len(suspicious),
            "suspicious_percentage": round(100.0 * len(suspicious) / len(results), 2) if results else 0.0,
            "suspicious_ids": [r["candidate_id"] for r in suspicious][:20],  # cap list for readability
        }
    except Exception as exc:
        logger.error("get_suspicious_count failed: %s", exc)
        return {"total_candidates": 0, "suspicious_count": 0, "suspicious_percentage": 0.0, "suspicious_ids": []}


def get_top_candidate_distribution(submission_df: pd.DataFrame) -> Dict[str, Any]:
    """Score distribution stats for the final ranked top-100."""
    try:
        if submission_df.empty:
            return {}
        return {
            "count": len(submission_df),
            "min_score": round(float(submission_df["score"].min()), 2),
            "max_score": round(float(submission_df["score"].max()), 2),
            "mean_score": round(float(submission_df["score"].mean()), 2),
            "std_score": round(float(submission_df["score"].std()), 2),
            "median_score": round(float(submission_df["score"].median()), 2),
        }
    except Exception as exc:
        logger.error("get_top_candidate_distribution failed: %s", exc)
        return {}


def run_evaluation(
    candidates_path: str = "data/candidates.jsonl",
    submission_path: str = "output/submission.csv",
    output_path: str = "analysis/evaluation_report.json",
) -> Dict[str, Any]:
    """
    Run full evaluation suite and save a JSON report.

    Returns:
        dict containing all computed evaluation metrics.
    """
    logger.info("=== HireIQ Evaluation Report ===")

    candidates = load_candidates(candidates_path)
    submission_df = load_submission(submission_path)

    report = {
        "total_candidates_in_dataset": len(candidates),
        "top_titles": get_top_titles(candidates),
        "top_skills": get_top_skills(candidates),
        "average_experience_years": get_avg_experience(candidates),
        "average_behavior_signals": get_avg_behavior_signals(candidates),
        "honeypot_summary": get_suspicious_count(candidates),
        "final_top_100_distribution": get_top_candidate_distribution(submission_df),
    }

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Evaluation report saved to %s", output_path)
    except Exception as exc:
        logger.error("Failed to save evaluation report: %s", exc)

    # Print human-readable summary to console
    print("\n=== HireIQ Pipeline Evaluation ===")
    print(f"Total candidates in dataset: {report['total_candidates_in_dataset']}")
    print(f"\nTop 10 titles: {report['top_titles']}")
    print(f"\nTop 15 skills: {report['top_skills']}")
    print(f"\nAverage experience: {report['average_experience_years']} years")
    print(f"\nAverage behavior signals: {report['average_behavior_signals']}")
    print(f"\nHoneypot summary: {report['honeypot_summary']}")
    print(f"\nFinal top-100 score distribution: {report['final_top_100_distribution']}")

    return report


if __name__ == "__main__":
    run_evaluation()