# HireIQ — System Architecture

## Overview
HireIQ is a candidate-ranking pipeline that scores and ranks candidates against a job description (JD) using six weighted signals, then demotes suspicious ("honeypot") profiles before generating a final, explainable submission.

## Module Map

| Module | File | Responsibility |
|---|---|---|
| Config | `src/config.py` | Central paths, `TOP_K`/`FINAL_TOP_K`, scoring weights |
| Utils | `src/utils.py` | JSONL/JSON I/O, candidate document builder, normalization |
| Semantic Retriever | `src/semantic_retriever.py` | Embeds candidates + JD with `all-MiniLM-L6-v2`, retrieves top-K via cosine similarity |
| Feature Extractor | `src/feature_extractor.py` | Computes skill, experience, title, behavior, profile-completeness scores |
| Honeypot Detector | `src/honeypot_detector.py` | Flags fabricated/unrealistic profiles (excess skills, implausible experience, etc.) -> suspicion score |
| Final Ranker | `src/final_ranker.py` | Combines weighted feature scores into a base score, applies honeypot penalty (capped at 90%), produces final ranked list |
| Reasoning Generator | `src/reasoning_generator.py` | Generates a one-line, honest (not purely positive) explanation per candidate based on which signals are strong/weak |
| Submission Generator | `src/submission_generator.py` | Validates ranked rows (no dupes, correct rank sequence, no empty reasoning) and writes `output/submission.csv` |
| Orchestrator | `run.py` | Wires all stages together end-to-end |
| Dashboard | `app.py` | Streamlit UI for exploring ranked candidates |

## Scoring Weights (from `src/final_ranker.py`)

| Signal | Weight |
|---|---|
| Semantic similarity | 35% |
| Skill overlap | 27% |
| Experience | 15% |
| Behavioral engagement | 10% |
| Title relevance | 8% |
| Profile completeness | 5% |

A honeypot penalty (multiplicative, capped at 90% reduction) is applied to the base score before final ranking.

## Data Flow

```
data/job_description.txt + data/candidates.jsonl
        |
        v
SemanticRetriever (src/semantic_retriever.py)
  - builds candidate documents (src/utils.py)
  - encodes with all-MiniLM-L6-v2
  - caches embeddings -> output/candidate_embeddings.npy, candidate_documents.json, candidate_ids.json
  - cosine similarity vs JD -> top TOP_K (500) candidates
        |
        v
Feature Extractor (src/feature_extractor.py)
  - skill / experience / title / behavior / profile scores
        |
        v
Honeypot Detector (src/honeypot_detector.py)
  - suspicion score per candidate
        |
        v
Final Ranker (src/final_ranker.py)
  - weighted base score x honeypot penalty
  - sorts, truncates to FINAL_TOP_K (100)
        |
        v
Reasoning Generator (src/reasoning_generator.py)
  - one-line explanation per candidate
        |
        v
Submission Generator (src/submission_generator.py)
  - validates rows
  - writes output/submission.csv
```

## Caching Strategy
Embeddings, candidate documents, and candidate IDs are cached to `output/` as `.npy`/`.json` files so repeated runs against the same `data/candidates.jsonl` skip re-encoding. These cache files are large and are excluded from the GitHub repo via `.gitignore`.

## Explainability
`reasoning_generator.py` deliberately includes caveat clauses for weak signals (e.g. low experience relative to the JD's stated minimum) rather than only praising candidates, so reviewers can spot borderline cases instead of seeing uniformly positive text.