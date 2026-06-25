# HireIQ — Redrob Hackathon Submission

> **Five-signal hybrid ranking system for the Redrob Senior AI Engineer JD**

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

HireIQ ranks candidates using five weighted signals:

| Signal | Weight | Source |
|---|---|---|
| Semantic similarity | 40% | Cosine similarity between JD and candidate embedding (all-MiniLM-L6-v2) |
| Skill overlap | 25% | JD keyword match with synonym normalization and partial credit |
| Experience | 15% | Years of experience tuned to JD sweet spot of 5–9 years |
| Behavioral engagement | 10% | Recruiter response rate, GitHub activity, interview completion, open-to-work flag |
| Title relevance | 10% | Fuzzy keyword match against 20+ engineering title variants |

**Design choices:**
- Penalizes keyword stuffers via title score
- Penalizes ghost candidates (low recruiter response, incomplete interviews) via behavior score
- Semantic context understanding helps down-rank consulting-background candidates
- Skill matching uses synonym normalization so `"pytorch"` matches `"PyTorch"`, etc.

---

## Pipeline

```
run.py
  │
  ├── Step 1  Load & parse job_description.txt → extract JD skills
  ├── Step 2  Semantic retrieval (top-500 from 100K candidates)
  ├── Step 3  Feature extraction  (feature_extractor.py)
  ├── Step 4  Hybrid ranking → top-100  (final_ranker.py)
  ├── Step 5  Reasoning generation  (reasoning_generator.py)
  └── Step 6  Write output/submission.csv  (submission_generator.py)
```

---

## Repository Structure

```
HireIQ/
├── run.py                        # Pipeline orchestrator
├── requirements.txt
├── submission_metadata.yaml
├── data/
│   ├── job_description.txt
│   └── sample_candidates.json    # 50-candidate dev sample
├── src/
│   ├── config.py
│   ├── semantic_retriever.py
│   ├── feature_extractor.py
│   ├── final_ranker.py
│   ├── reasoning_generator.py
│   ├── submission_generator.py
│   └── utils.py
└── output/
    └── submission.csv            # Final ranked output
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
#    Copy candidates.jsonl.gz into data/

# 4. Run
python run.py
```

Output is written to `output/submission.csv`.

**Runtime:** ~12 minutes on first run (embedding generation + caching), ~2 minutes on subsequent runs.

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
| `score` | Weighted composite score (0–100) |
| `reason` | One-line human-readable explanation |

---

## AI Tools

Claude (Anthropic) was used for code review, debugging, and architectural discussion during development. No candidate data was fed to any LLM. All scoring weights, feature engineering decisions, and ranking logic were made by the team.