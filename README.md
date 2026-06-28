# HireIQ — Candidate Intelligence Pipeline

> **Six-signal hybrid ranking system with fraud detection, for the Redrob Senior AI Engineer JD**

---

## Team

| Name | Role | Email | Phone |
|---|---|---|---|
| **Dakshyani Ashok Borade** | **Team Leader · AI / ML** | dakshyaniborade@gmail.com | — |
| Ishika Vilasrao Mohod *(primary contact)* | Data Science | mohodishika2022@gmail.com | +91-9028638572 |
| Ashwini Umakant Koturwar | MERN Stack Developer | koturwarashwini1@gmail.com | — |

---

## Links

| Resource | URL |
|---|---|
| GitHub | https://github.com/Cheriex777/HireIQ.git |
| Colab sandbox | https://colab.research.google.com/drive/1veFIu33Nxd3j3Mw4zUuQqkKsrpaxkYL1?usp=sharing |

---

## Approach

### Six Ranking Signals

| # | Signal | Weight | Source |
|---|---|---|---|
| 1 | **Semantic similarity** | 35% | Cosine similarity between JD and candidate embedding (`all-MiniLM-L6-v2`), built from headline, summary, title, skills, career history, and education |
| 2 | **Skill overlap** | 27% | JD keyword match with synonym normalization and partial credit |
| 3 | **Experience** | 15% | Years of experience tuned to JD sweet spot of 5–9 years |
| 4 | **Behavioral engagement** | 10% | Recruiter response rate, GitHub activity, interview completion, open-to-work flag |
| 5 | **Title relevance** | 8% | Fuzzy keyword match against 20+ engineering title variants |
| 6 | **Profile completeness** | 5% | Rewards fully-filled profiles (headline, summary, skills, career history, education) |

### Scoring Formula

```
final_score = base_score × (1 − honeypot_penalty)
```

### Honeypot / Fraud Detection

Every candidate is scanned for suspicious patterns before ranking:

- Unrealistic years of experience (> 30 years)
- Excessive or padded skill lists (> 25 skills)
- Incomplete profiles with senior-sounding titles
- Senior-title-with-zero-activity mismatches
- Abnormally perfect behavioral signals
- Inconsistent career history

Flagged profiles receive a **multiplicative score penalty** rather than outright rejection, which avoids false-positive eliminations while still down-ranking suspicious entries.

- Validated against synthetic fake profiles → correctly scored 0.91 / 1.0 suspicion
- **0 suspicious profiles found** in the real 100K dataset — confirms genuine, fraud-free data

### Design Choices

- Penalizes **keyword stuffers** via title score and honeypot excessive-skills check
- Penalizes **ghost candidates** (low recruiter response, incomplete interviews) via behavior score
- Semantic context (career history + education included in embedding) helps down-rank candidates whose only AI exposure is recent LangChain / wrapper projects
- Skill matching uses synonym normalization so `"FAISS"`, `"vector db"`, and `"vector search"` all resolve correctly

---

## Pipeline

```
run.py
  │
  ├── Step 1    Load & parse data/job_description.txt → extract JD skills
  ├── Step 2    Semantic retrieval — top-500 from 100K candidates
  │               src/retrieval/build_documents.py
  │               src/retrieval/generate_embeddings.py
  │               src/retrieval/retrieve_candidates.py
  ├── Step 3    Feature extraction        (src/feature_extractor.py)
  ├── Step 3.5  Honeypot / fraud detection (src/honeypot_detector.py)
  ├── Step 4    Hybrid ranking → top-100   (src/final_ranker.py)
  ├── Step 5    Reasoning generation       (src/reasoning_generator.py)
  └── Step 6    Write output/submission.csv (src/submission_generator.py)
```

See [`docs/architecture.md`](docs/architecture.md) and [`docs/project_flow.md`](docs/project_flow.md) for detailed design notes and the full architecture diagram.

---

## Repository Structure

```
HireIQ/
├── README.md
├── LICENSE
├── requirements.txt
├── submission_metadata.yaml
├── .gitignore
│
├── run.py                              # Pipeline orchestrator
├── app.py                              # Streamlit dashboard
│
├── src/
│   ├── __init__.py
│   ├── config.py                       # Paths, weights, and shared constants
│   ├── utils.py                        # Shared helpers (I/O, normalization, embeddings)
│   ├── semantic_retriever.py           # Top-level retrieval coordinator
│   ├── feature_extractor.py            # Six-signal feature computation
│   ├── honeypot_detector.py            # Fraud / anomaly detection
│   ├── final_ranker.py                 # Weighted hybrid ranking
│   ├── reasoning_generator.py          # Human-readable explanation per candidate
│   ├── submission_generator.py         # Writes submission.csv
│   └── retrieval/
│       ├── build_documents.py          # Build text documents from candidate records
│       ├── generate_embeddings.py      # Encode documents → numpy embeddings cache
│       └── retrieve_candidates.py      # Cosine similarity search → top-K candidates
│
├── data/
│   ├── job_description.txt             # Target JD (Redrob Senior AI Engineer)
│   └── sample_candidates.json          # Small sample for testing (full 100K not committed)
│
├── output/
│   └── submission.csv                  # Final ranked output — top 100 candidates
│
├── tests/
│   ├── test_feature_extract.py         # Unit tests for feature_extractor.py
│   └── test_final_rank.py              # Unit tests for final_ranker.py
│
└── docs/
    ├── architecture.md                 # Module design and signal explanations
    ├── project_flow.md                 # End-to-end pipeline walkthrough
    ├── architecture_diagram.png        # Visual system diagram
    └── screenshots/
        ├── 01_dashboard_overview.png   # KPIs · score distribution · fit band donut
        ├── 02_score_distribution.png   # Score histogram + top candidates bar chart
        ├── 03_top_candidates_chart.png # Top-10 candidates bar chart (zoomed)
        ├── 04_card_view.png            # Leaderboard card view with reasoning
        └── 05_table_view.png           # Leaderboard table view with filters
```

---

## Setup & Reproduce

### 1 — Clone and install

```bash
git clone https://github.com/Cheriex777/HireIQ.git
cd HireIQ
pip install -r requirements.txt
```

### 2 — Add the full dataset

Place the full `candidates.jsonl` (100K records) into `data/`.
The committed `data/sample_candidates.json` can be used for a quick smoke-test.

### 3 — Run the pipeline

```bash
python run.py
```

Output is written to `output/submission.csv`.

**Runtime:**
- First run (embedding generation for 100K candidates): ~2 hours, one-time cost
- All subsequent runs (embeddings loaded from cache): **~25–30 seconds**

### 4 — Run the tests

```bash
pytest tests/
```

### 5 — Launch the dashboard

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

Upload `output/submission.csv` using the sidebar uploader to explore results interactively.

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

---

## Final Results

```
Dataset evaluated            :  100,000 candidates
Final submission size        :  top 100
──────────────────────────────────────────────────
Top score                    :  75.91
Mean score                   :  66.11
Median score                 :  64.94
Std deviation                :  3.66
──────────────────────────────────────────────────
Suspicious profiles flagged  :  0 / 100,000
```

---

## Dashboard (`app.py`)

The Streamlit dashboard lets you explore ranked results interactively after uploading `submission.csv`.

### Sidebar Controls

| Control | Description |
|---|---|
| **Upload submission.csv** | Drag-and-drop or browse to load your pipeline output |
| **Use sample data** | Toggle to load the built-in 50-candidate sample without uploading |
| **Score range slider** | Filter candidates by minimum / maximum score |
| **Show top N candidates** | Numeric stepper to control how many candidates are shown (default 10) |
| **Search candidate ID** | Jump directly to a specific `CAND_XXXXXX` entry |
| **Download CSV** | Export the currently filtered view as a CSV file |

### Main Panels

| Panel | Description |
|---|---|
| **Summary metrics** | Total candidates, candidates currently shown, top score, average score, and strong-fit count (≥ 55) as headline KPIs |
| **Score distribution** | Histogram of final scores with dashed threshold lines marking Good fit and Strong fit bands |
| **Fit band breakdown** | Donut chart showing proportion of candidates in each fit category (100% Strong fit on real dataset) |
| **Top candidates by score** | Bar chart of the top-N highest-scoring candidates |
| **Leaderboard — Card view** | Candidate cards showing rank badge, score, fit label, and one-line reasoning; selected card highlighted |
| **Leaderboard — Table view** | Sortable table with Rank, Candidate ID, Score, and full Reasoning columns |

### Screenshots

**1 · Dashboard Overview — KPIs · Score Distribution · Fit Band Breakdown**
![Dashboard Overview](docs/screenshots/01_dashboard_overview.png)

**2 · Score Distribution + Top Candidates Bar Chart**
![Score Distribution and Chart](docs/screenshots/02_score_distribution.png)

**3 · Top-10 Candidates Bar Chart (zoomed)**
![Top Candidates Chart](docs/screenshots/03_top_candidates_chart.png)

**4 · Leaderboard — Card View**
![Card View](docs/screenshots/04_card_view.png)

**5 · Leaderboard — Table View with Filters**
![Table View](docs/screenshots/05_table_view.png)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## AI Tools

Claude (Anthropic) was used for code review, debugging, and architectural discussion during development. No candidate data was fed to any LLM. All scoring weights, feature engineering decisions, and ranking logic were made by the team.