import json
from sentence_transformers import SentenceTransformer

print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Reading documents...")

with open(
    "output/candidate_documents.json",
    "r",
    encoding="utf-8"
) as f:

    documents = json.load(f)

documents = documents[:1000]

texts = [doc["text"] for doc in documents]

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Shape:", embeddings.shape)


