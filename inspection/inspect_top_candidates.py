import json

top_ids = [
    "CAND_0000031",
    "CAND_0000981",
    "CAND_0000273",
    "CAND_0000388",
    "CAND_0000705"
]

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:

    for line in f:

        candidate = json.loads(line)

        if candidate["candidate_id"] in top_ids:

            print("\n" + "="*80)

            print("ID:", candidate["candidate_id"])

            print("TITLE:",
                  candidate["profile"]["current_title"])

            print()

            print("YEARS:",
                  candidate["profile"]["years_of_experience"])

            print()

            print("HEADLINE:")
            print(candidate["profile"]["headline"])

            print()

            print("RECRUITER RESPONSE:",
                  candidate["redrob_signals"]["recruiter_response_rate"])

            print()

            print("GITHUB:",
                  candidate["redrob_signals"]["github_activity_score"])

            print()

            print("SAVED BY RECRUITERS:",
                  candidate["redrob_signals"]["saved_by_recruiters_30d"])

            print()

            print("SEARCH APPEARANCE:",
                  candidate["redrob_signals"]["search_appearance_30d"])