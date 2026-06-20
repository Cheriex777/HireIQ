import json

keywords = [
    "recommendation",
    "retrieval",
    "search",
    "ranking",
    "embedding",
    "vector"
]

matches = []

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        candidate = json.loads(line)

        text = (
            candidate["profile"]["headline"] + " " +
            candidate["profile"]["summary"]
        ).lower()

        score = sum(
            keyword in text
            for keyword in keywords
        )

        if score >= 3:
            matches.append(candidate)

print("Found:", len(matches))

for c in matches[:20]:
    print()
    print("ID:", c["candidate_id"])
    print("Title:", c["profile"]["current_title"])
    print("Headline:", c["profile"]["headline"])