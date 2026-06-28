# HireIQ — Project Flow (End-to-End Run)

This describes exactly what happens when `run.py` is executed.

## Step 1 — Load Inputs
- `load_job_description()` reads `data/job_description.txt`.
- `parse_jd_skills()` extracts the "Key skills:" line from the JD (falls back to a curated list if absent).

## Step 2 — Semantic Retrieval
- `SemanticRetriever` (`src/semantic_retriever.py`) loads `all-MiniLM-L6-v2`.
- `load_and_encode_candidates()` reads `data/candidates.jsonl`, builds a text document per candidate (`build_candidate_document` in `src/utils.py`), and encodes all candidates.
- Encoded embeddings/documents/IDs are cached to `output/` so re-runs are fast unless `force_recompute=True`.
- `retrieve()` computes cosine similarity between the JD embedding and all candidate embeddings, returning the top `TOP_K` (500) candidates.

## Step 3 — Feature Engineering
- `extract_features_batch()` (`src/feature_extractor.py`) computes, per candidate:
  - Skill overlap score
  - Experience score
  - Title relevance score (via `TITLE_SCORE_MAP`)
  - Behavioral engagement score
  - Profile completeness score

## Step 4 — Honeypot Detection
- `detect_honeypots_batch()` (`src/honeypot_detector.py`) runs suspicion checks per candidate:
  - Unrealistic experience (negative or > 30 years)
  - Excessive/padded skill lists (> 25 skills)
  - Other inconsistency checks
- Produces a 0–1 suspicion score and corresponding penalty.

## Step 5 — Hybrid Ranking
- `HybridRanker` (`src/final_ranker.py`) combines the six feature scores using `DEFAULT_WEIGHTS`, applies the honeypot penalty multiplicatively (capped at `MAX_PENALTY_CAP = 0.9`), and sorts candidates by final score.
- The list is truncated to `FINAL_TOP_K` (100).

## Step 6 — Reasoning Generation
- `generate_reasoning_batch()` (`src/reasoning_generator.py`) builds a one-line explanation per candidate, citing only the signals that are genuinely strong, and adding a caveat clause when a signal is below threshold (e.g. fewer years of experience than the JD's stated minimum).

## Step 7 — Submission Generation
- `generate_submission()` (`src/submission_generator.py`) validates the final rows:
  - Row count matches expected count
  - No duplicate `candidate_id`s
  - Ranks form a clean 1..N sequence
  - Scores are sorted descending
  - No empty reasoning text
- Writes the validated rows to `output/submission.csv` with columns: `candidate_id, rank, score, reasoning`.

## Step 8 — Inspection (optional, manual)
- `app.py` (Streamlit) can be run separately (`streamlit run app.py`) to visually explore the ranked output in `output/submission.csv`.

## Failure Behavior
The pipeline is designed to fail loudly: `validate_submission_rows()` raises a `ValueError` listing every issue found if the ranked data is malformed, rather than silently writing a broken `submission.csv`.