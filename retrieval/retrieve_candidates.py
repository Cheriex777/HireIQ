import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading candidate documents...")

with open("output/candidate_documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

documents = documents[:1000]

candidate_texts = [doc["text"] for doc in documents]

print("Generating candidate embeddings...")

candidate_embeddings = model.encode(
    candidate_texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

job_description = """
Senior AI Engineer

Must have:
LLMs
RAG
Embeddings
Vector Search
Information Retrieval
Ranking Systems
Recommendation Systems
FAISS
Production ML Systems

5+ years experience
"""

print("Embedding JD...")

jd_embedding = model.encode(
    [job_description],
    convert_to_numpy=True
)

scores = cosine_similarity(
    jd_embedding,
    candidate_embeddings
)[0]

top_indices = np.argsort(scores)[::-1][:20]

print("\nTOP 20 CANDIDATES\n")

for idx in top_indices:

    doc = documents[idx]

    print(
        round(scores[idx],4),
        doc["candidate_id"]
    )