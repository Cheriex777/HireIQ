import json
from collections import Counter

counter = Counter()

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:

    for line in f:

        candidate = json.loads(line)

        title = candidate["profile"]["current_title"]

        counter[title] += 1

print("\nTOP 50 TITLES\n")

for title, count in counter.most_common(50):

    print(title, ":", count)