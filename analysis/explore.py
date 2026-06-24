import json
from collections import Counter

title_counter = Counter()

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        candidate = json.loads(line)

        title = candidate["profile"]["current_title"]
        title_counter[title] += 1

print("\nTop 20 Job Titles:\n")

for title, count in title_counter.most_common(20):
    print(f"{title}: {count}")