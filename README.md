# HireIQ — Redrob Hackathon Submission

> **Six-signal hybrid ranking system with fraud detection, for the Redrob Senior AI Engineer JD**

---

## Team

| Name | Role | Email |
|---|---|---|
| Ishika Vilasrao Mohod | Data Science | mohodishika2022@gmail.com |
| Ashwini Umakant Koturwar | MERN Stack Developer | koturwarashwini1@gmail.com |
| Dakshyani Ashok Borade | AI / ML | dakshyaniborade@gmail.com |

**Primary contact:** Ishika Vilasrao Mohod — +91-9028638572

---

## Links

| Resource | URL |
|---|---|
| GitHub | https://github.com/Cheriex777/HireIQ.git |
| Colab sandbox | https://colab.research.google.com/drive/1veFIu33Nxd3j3Mw4zUuQqkKsrpaxkYL1?usp=sharing |

---

## Approach

HireIQ ranks candidates using **six weighted signals**, then applies a fraud-detection penalty:

| Signal | Weight | Source |
|---|---|---|
| Semantic similarity | 35% | Cosine similarity between JD and candidate embedding (all-MiniLM-L6-v2), built from headline, summary, title, skills, career history, and education |
| Skill overlap | 27% | JD keyword match with synonym normalization and partial credit |
| Experience | 15% | Years of experience tuned to JD sweet spot of 5–9 years |
| Behavioral engagement | 10% | Recruiter response rate, GitHub activity, interview completion, open-to-work flag |
| Title relevance | 8% | Fuzzy keyword match against 20+ engineering title variants |
| Profile completeness | 5% | Rewards fully-filled profiles (headline, summary, skills, career history, education) |

```
final_score = base_score × (1 − honeypot_penalty)
```

**Honeypot / Fraud Detection:** Every candidate is scanned for suspicious patterns before ranking — unrealistic experience, excessive/padded skill lists, incomplete profiles, senior-title-with-zero-activity mismatches, abnormally perfect behavioral signals, and inconsistent career history. Flagged profiles get a multiplicative score penalty rather than outright rejection, avoiding false-positive demos.

- Validated against synthetic fake profiles (correctly scored 0.91/1.0 suspicion)
- **0 suspicious profiles found** in the real 100K dataset — confirms genuine, fraud-free data

**Design choices:**
- Penalizes keyword stuffers via title score and honeypot's excessive-skills check
- Penalizes ghost candidates (low recruiter response, incomplete interviews) via behavior score
- Semantic context (now including career history + education) helps down-rank candidates whose only AI exposure is recent LangChain/wrapper projects
- Skill matching uses synonym normalization so `"FAISS"`, `"vector db"`, and `"vector search"` all resolve correctly

---

## Pipeline

```
run.py
  │
  ├── Step 1   Load & parse job_description.txt → extract JD skills
  ├── Step 2   Semantic retrieval (top-500 from 100K candidates)
  ├── Step 3   Feature extraction        (feature_extractor.py)
  ├── Step 3.5 Honeypot / fraud detection (honeypot_detector.py)
  ├── Step 4   Hybrid ranking → top-100   (final_ranker.py)
  ├── Step 5   Reasoning generation       (reasoning_generator.py)
  └── Step 6   Write output/submission.csv (submission_generator.py)
```


## Repository Structure

```
HireIQ/
├── run.py                        # Pipeline orchestrator
├── requirements.txt
├── submission_metadata.yaml
├── data/
│   ├── job_description.txt
│   └── candidates.jsonl          # Full 100K dataset
├── src/
│   ├── config.py
│   ├── semantic_retriever.py
│   ├── feature_extractor.py
│   ├── honeypot_detector.py
│   ├── final_ranker.py
│   ├── reasoning_generator.py
│   ├── submission_generator.py
│   └── utils.py
├── analysis/
│   ├── evaluate_pipeline.py
│   └── evaluation_report.json    # Dataset-level stats (titles, skills, experience, fraud scan)
└── output/
    └── submission.csv            # Final ranked output (top 100)
```

---

## Setup & Reproduce

```bash
# 1. Clone
git clone https://github.com/Cheriex777/HireIQ.git
cd HireIQ

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the full dataset
#    Copy candidates.jsonl into data/

# 4. Run the pipeline
python run.py

# 5. (Optional) Generate dataset-level evaluation stats
python analysis/evaluate_pipeline.py
```

Output is written to `output/submission.csv`.

**Runtime:**
- First run (full embedding generation for 100K candidates): ~2 hours, one-time cost
- All subsequent runs (embeddings cached to `output/`): **~25–30 seconds**

---

## Environment

| Property | Value |
|---|---|
| OS | Windows 11 |
| Python | 3.12.4 |
| CPU cores | 4 |
| RAM | 16 GB |
| GPU | None (CPU inference) |
| Network during ranking | No |

---

## Output Format

`output/submission.csv` — one row per candidate:

| Column | Description |
|---|---|
| `candidate_id` | Candidate identifier |
| `rank` | Final rank (1 = best) |
| `score` | Final weighted + fraud-adjusted score (0–100) |
| `reasoning` | One-line human-readable explanation |

**Final results (100K dataset, top 100 candidates):**
- Score range: 61.71 – 75.91
- Mean score: 66.11 · Median: 64.94 · Std dev: 3.66
- Suspicious/fraudulent profiles flagged: 0 / 100,000

---

## AI Tools

Claude (Anthropic) was used for code review, debugging, and architectural discussion during development. No candidate data was fed to any LLM. All scoring weights, feature engineering decisions, and ranking logic were made by the team.