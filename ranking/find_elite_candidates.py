import json

elite_titles = [
    "Recommendation Systems Engineer",
    "Search Engineer",
    "Senior NLP Engineer",
    "Senior AI Engineer",
    "Lead AI Engineer",
    "Staff Machine Learning Engineer"
]

count = 0

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:

    for line in f:

        candidate = json.loads(line)

        title = candidate["profile"]["current_title"]

        if title in elite_titles:

            count += 1

            print("\n" + "=" * 60)
            print("ID:", candidate["candidate_id"])
            print("TITLE:", title)
            print(
                "YEARS:",
                candidate["profile"]["years_of_experience"]
            )

print("\n" + "=" * 60)
print("TOTAL ELITE CANDIDATES FOUND:", count)