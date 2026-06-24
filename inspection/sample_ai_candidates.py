import json

keywords = [
    "recommendation",
    "retrieval",
    "search",
    "ranking",
    "embedding",
    "vector"
]

count = 0

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        candidate = json.loads(line)

        text = (
            candidate["profile"]["headline"] + " " +
            candidate["profile"]["summary"]
        ).lower()

        if any(keyword in text for keyword in keywords):

            print("\nID:", candidate["candidate_id"])
            print("Title:", candidate["profile"]["current_title"])
            print("Headline:", candidate["profile"]["headline"])

            count += 1

            if count >= 20:
                break