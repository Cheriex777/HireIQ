import json
from collections import Counter

skill_counter = Counter()

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        candidate = json.loads(line)

        for skill in candidate["skills"]:
            skill_counter[skill["name"]] += 1

print("\nTop 50 Skills:\n")

for skill, count in skill_counter.most_common(50):
    print(f"{skill}: {count}")
    