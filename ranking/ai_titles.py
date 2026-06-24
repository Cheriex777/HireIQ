import json
from collections import Counter

keywords = [
    "ai",
    "machine learning",
    "ml",
    "nlp",
    "data scientist",
    "recommendation",
    "search"
]

counter = Counter()

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        candidate = json.loads(line)

        title = candidate["profile"]["current_title"].lower()

        for keyword in keywords:
            if keyword in title:
                counter[keyword] += 1

print(counter)