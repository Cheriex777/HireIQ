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

            print("\n" + "=" * 80)
            print("ID:", candidate["candidate_id"])
            print("TITLE:", candidate["profile"]["current_title"])
            print("YEARS:", candidate["profile"]["years_of_experience"])

            print("\nHEADLINE:")
            print(candidate["profile"]["headline"])

            print("\nSUMMARY:")
            print(candidate["profile"]["summary"])

            print("\nSKILLS:")
            for skill in candidate["skills"][:15]:
                print("-", skill["name"])