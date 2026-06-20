import json
from collections import Counter

target_skills = [
    "NLP",
    "RAG",
    "Embeddings",
    "Vector Search",
    "Milvus",
    "FAISS",
    "Recommendation Systems",
    "Information Retrieval",
    "LLMs",
    "Fine-tuning LLMs",
    "Semantic Search",
    "Ranking Systems"
]

counter = Counter()

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        candidate = json.loads(line)

        candidate_skills = {
            skill["name"] for skill in candidate["skills"]
        }

        for skill in target_skills:
            if skill in candidate_skills:
                counter[skill] += 1

print("\nTarget Skills:\n")

for skill, count in counter.items():
    print(f"{skill}: {count}")