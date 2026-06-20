import json

target_id = "CAND_0008295"

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:

    for line in f:

        candidate = json.loads(line)

        if candidate["candidate_id"] == target_id:

            print("\nTITLE:")
            print(candidate["profile"]["current_title"])

            print("\nHEADLINE:")
            print(candidate["profile"]["headline"])

            print("\nSUMMARY:")
            print(candidate["profile"]["summary"])

            print("\nSKILLS:")

            for skill in candidate["skills"]:
                print("-", skill["name"])

            break