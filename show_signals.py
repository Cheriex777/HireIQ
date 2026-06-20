import json

candidate_ids = [
    "CAND_0000031",
    "CAND_0002025",
    "CAND_0005260"
]

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        candidate = json.loads(line)

        if candidate["candidate_id"] in candidate_ids:

            signals = candidate["redrob_signals"]

            print("\n" + "=" * 80)
            print("ID:", candidate["candidate_id"])
            print("TITLE:", candidate["profile"]["current_title"])

            print("\nBehavioral Signals")
            print("-------------------")

            print(
                "Profile Completeness:",
                signals["profile_completeness_score"]
            )

            print(
                "Recruiter Response Rate:",
                signals["recruiter_response_rate"]
            )

            print(
                "GitHub Activity:",
                signals["github_activity_score"]
            )

            print(
                "Interview Completion:",
                signals["interview_completion_rate"]
            )

            print(
                "Saved By Recruiters:",
                signals["saved_by_recruiters_30d"]
            )

            print(
                "Search Appearance:",
                signals["search_appearance_30d"]
            )

            print(
                "Offer Acceptance:",
                signals["offer_acceptance_rate"]
            )

            print(
                "Open To Work:",
                signals["open_to_work_flag"]
            )

            print(
                "Verified Email:",
                signals["verified_email"]
            )

            print(
                "Verified Phone:",
                signals["verified_phone"]
            )