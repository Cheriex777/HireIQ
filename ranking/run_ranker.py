"""
run_ranker.py — HireIQ
Run this ONE file to produce submission.csv

Usage:
    python ranking/run_ranker.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import config
from final_ranker        import load_candidates, rank_candidates
from reasoning_generator import attach_reasoning
from submission_generator import write_submission


def main():
    print("=" * 60)
    print("HireIQ — AI Ranking Engine")
    print("=" * 60)

    # 1. Load candidates
    candidates = load_candidates()

    # 2. Load job description
    jd_text = open(config.JD_PATH, encoding="utf-8").read()
    print(f"Job description loaded ({len(jd_text)} characters)")

    # 3. Rank candidates
    results = rank_candidates(jd_text, candidates, top_n=config.TOP_N)

    # 4. Generate reasoning
    print("Generating reasoning for each candidate...")
    attach_reasoning(results)

    # 5. Write submission CSV
    write_submission(results)

    print("\nDone! Your submission.csv is ready in the output/ folder.")


if __name__ == "__main__":
    main()
