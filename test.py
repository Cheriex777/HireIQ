import json

candidates = []

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        candidates.append(json.loads(line))

print("Total candidates:", len(candidates))

print("First candidate ID:", candidates[0]["candidate_id"])

print("First candidate title:", candidates[0]["profile"]["current_title"])