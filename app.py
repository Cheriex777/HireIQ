"""
app.py — HireIQ Candidate Intelligence Dashboard
Run: streamlit run app.py
"""

import io
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HireIQ — Candidate Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #58a6ff !important;
    font-size: 2rem !important;
    font-weight: 700;
}

/* ── Rank badge ── */
.rank-badge {
    display: inline-block;
    background: #1f2937;
    color: #58a6ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    border: 1px solid #30363d;
}

/* ── Score pill ── */
.score-high  { color: #3fb950; font-weight: 700; }
.score-mid   { color: #d29922; font-weight: 700; }
.score-low   { color: #f85149; font-weight: 700; }

/* ── Section header ── */
.section-header {
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 8px;
    border-bottom: 1px solid #21262d;
    padding-bottom: 4px;
}

/* ── Candidate card ── */
.cand-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
    transition: border-color 0.15s;
}
.cand-card:hover { border-color: #58a6ff; }
.cand-card .cand-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 600;
    color: #e6edf3;
}
.cand-card .cand-reason {
    color: #8b949e;
    font-size: 0.88rem;
    margin-top: 6px;
    line-height: 1.5;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #21262d !important;
    border-radius: 8px;
}

/* ── Divider ── */
hr { border-color: #21262d; }

/* ── Plotly charts transparent bg ── */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

SAMPLE_CSV = """candidate_id,rank,score,reasoning
CAND_0088025,1,74.61,"8 years of strong relevant experience, good semantic relevance to job description, high recruiter engagement, currently a Staff Machine Learning Engineer."
CAND_0071974,2,73.94,"7 years of strong relevant experience, good semantic relevance to job description, high recruiter engagement, currently a Senior AI Engineer."
CAND_0055905,3,73.92,"8 years of strong relevant experience, good semantic relevance to job description, moderate recruiter engagement, currently a Senior Machine Learning Engineer."
CAND_0002025,4,73.45,"5 years of relevant experience, good semantic relevance to job description, high recruiter engagement, currently a Senior AI Engineer."
CAND_0077337,5,73.3,"7 years of strong relevant experience, good semantic relevance to job description, high recruiter engagement, currently a Staff Machine Learning Engineer."
CAND_0081846,6,72.77,"6 years of strong relevant experience, good semantic relevance to job description, high recruiter engagement, currently a Lead AI Engineer."
CAND_0044855,7,72.38,"6 years of strong relevant experience, good semantic relevance to job description, moderate recruiter engagement."
CAND_0046525,8,72.15,"6 years of strong relevant experience, good semantic relevance to job description, high recruiter engagement, currently a Senior Machine Learning Engineer."
CAND_0008425,9,70.89,"7 years of strong relevant experience, good semantic relevance to job description, high recruiter engagement, currently a Senior NLP Engineer."
CAND_0007411,10,70.2,"8 years of strong relevant experience, good semantic relevance to job description, moderate recruiter engagement, currently a Senior Machine Learning Engineer."
"""


def score_color(score: float) -> str:
    if score >= 55:
        return "score-high"
    elif score >= 40:
        return "score-mid"
    return "score-low"


def score_band(score: float) -> str:
    if score >= 55:
        return "Strong fit"
    elif score >= 40:
        return "Good fit"
    elif score >= 30:
        return "Moderate fit"
    return "Weak fit"


def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    # Normalise column names
    rename = {}
    for col in df.columns:
        if "reason" in col:
            rename[col] = "reasoning"
        elif col == "score":
            rename[col] = "score"
    df = df.rename(columns=rename)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(0).astype(int)
    return df.sort_values("rank").reset_index(drop=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎯 HireIQ")
    st.markdown("<p style='color:#8b949e;font-size:0.82rem;'>Candidate Intelligence Dashboard</p>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded = st.file_uploader("Upload submission.csv", type=["csv"])
    use_sample = st.checkbox("Use sample data (50 candidates)", value=uploaded is None)

    st.markdown("---")
    st.markdown("<p class='section-header'>Filters</p>", unsafe_allow_html=True)

    if uploaded or use_sample:
        src = uploaded if uploaded else io.StringIO(SAMPLE_CSV)
        df_raw = load_csv(src)

        score_min, score_max = float(df_raw["score"].min()), float(df_raw["score"].max())
        score_range = st.slider(
            "Score range",
            min_value=math.floor(score_min),
            max_value=math.ceil(score_max),
            value=(math.floor(score_min), math.ceil(score_max)),
        )
        rank_top = st.number_input("Show top N candidates", min_value=5, max_value=len(df_raw), value=min(20, len(df_raw)), step=5)
        search_id = st.text_input("Search candidate ID", placeholder="CAND_000…")

        st.markdown("---")
        # Download button
        csv_bytes = df_raw.to_csv(index=False).encode()
        st.download_button(
            label="⬇ Download CSV",
            data=csv_bytes,
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        df_raw = pd.DataFrame()
        score_range = (0, 100)
        rank_top = 20
        search_id = ""

    st.markdown("---")
    st.markdown("<p style='color:#8b949e;font-size:0.75rem;'>Team HireIQ · Redrob Hackathon</p>", unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────

st.markdown("## Candidate Intelligence Dashboard")
st.markdown("<p style='color:#8b949e;margin-top:-12px;'>Redrob · Senior AI Engineer · Hybrid five-signal ranking</p>", unsafe_allow_html=True)

if df_raw.empty:
    st.info("Upload a `submission.csv` in the sidebar or enable sample data to get started.")
    st.stop()

# Apply filters
df = df_raw[
    (df_raw["score"] >= score_range[0]) &
    (df_raw["score"] <= score_range[1])
].head(int(rank_top))

if search_id.strip():
    df = df[df["candidate_id"].str.contains(search_id.strip(), case=False)]

# ── KPI row ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total candidates", len(df_raw))
k2.metric("Showing", len(df))
k3.metric("Top score", f"{df_raw['score'].max():.2f}")
k4.metric("Avg score", f"{df_raw['score'].mean():.2f}")
strong = len(df_raw[df_raw["score"] >= 55])
k5.metric("Strong fits (≥55)", strong)

st.markdown("---")

# ── Charts row ───────────────────────────────────────────────────────────────
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("<p class='section-header'>Score distribution</p>", unsafe_allow_html=True)
    fig_hist = px.histogram(
        df_raw, x="score", nbins=20,
        color_discrete_sequence=["#58a6ff"],
        template="plotly_dark",
    )
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        height=260,
        bargap=0.1,
        xaxis=dict(title="Score", color="#8b949e", gridcolor="#21262d"),
        yaxis=dict(title="Count", color="#8b949e", gridcolor="#21262d"),
        font=dict(family="Inter", color="#8b949e"),
        showlegend=False,
    )
    fig_hist.add_vline(x=55, line_dash="dash", line_color="#3fb950", annotation_text="Strong fit", annotation_font_color="#3fb950")
    fig_hist.add_vline(x=40, line_dash="dash", line_color="#d29922", annotation_text="Good fit", annotation_font_color="#d29922")
    st.plotly_chart(fig_hist, use_container_width=True)

with col_chart2:
    st.markdown("<p class='section-header'>Fit band breakdown</p>", unsafe_allow_html=True)
    band_counts = df_raw["score"].apply(score_band).value_counts().reset_index()
    band_counts.columns = ["band", "count"]
    color_map = {
        "Strong fit": "#3fb950",
        "Good fit": "#58a6ff",
        "Moderate fit": "#d29922",
        "Weak fit": "#f85149",
    }
    fig_pie = px.pie(
        band_counts, names="band", values="count",
        color="band", color_discrete_map=color_map,
        template="plotly_dark",
        hole=0.55,
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        height=260,
        font=dict(family="Inter", color="#8b949e"),
        legend=dict(font=dict(color="#8b949e")),
        showlegend=True,
    )
    fig_pie.update_traces(textfont_color="white")
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Score bar chart ───────────────────────────────────────────────────────────
st.markdown("<p class='section-header'>Top candidates by score</p>", unsafe_allow_html=True)

top_bar = df.head(20).copy()
top_bar["color"] = top_bar["score"].apply(
    lambda s: "#3fb950" if s >= 55 else ("#58a6ff" if s >= 40 else "#d29922")
)
fig_bar = go.Figure(go.Bar(
    x=top_bar["candidate_id"],
    y=top_bar["score"],
    marker_color=top_bar["color"],
    text=top_bar["score"].apply(lambda s: f"{s:.1f}"),
    textposition="outside",
    textfont=dict(color="#e6edf3", size=11),
))
fig_bar.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=10, b=60),
    height=300,
    xaxis=dict(tickangle=-35, color="#8b949e", gridcolor="#21262d", tickfont=dict(family="JetBrains Mono", size=10)),
    yaxis=dict(color="#8b949e", gridcolor="#21262d", range=[0, top_bar["score"].max() * 1.18]),
    font=dict(family="Inter", color="#8b949e"),
    showlegend=False,
)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ── Leaderboard table ─────────────────────────────────────────────────────────
st.markdown("<p class='section-header'>Leaderboard</p>", unsafe_allow_html=True)

tab_cards, tab_table = st.tabs(["Card view", "Table view"])

with tab_cards:
    for _, row in df.iterrows():
        sc = row["score"]
        cls = score_color(sc)
        band = score_band(sc)
        reasoning = str(row.get("reasoning", "—"))

        st.markdown(f"""
        <div class="cand-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span class="cand-id">{row['candidate_id']}</span>
                <span>
                    <span class="rank-badge">#{int(row['rank'])}</span>
                    &nbsp;
                    <span class="{cls}">{sc:.2f}</span>
                    &nbsp;
                    <span style="color:#8b949e;font-size:0.8rem;">· {band}</span>
                </span>
            </div>
            <div class="cand-reason">{reasoning}</div>
        </div>
        """, unsafe_allow_html=True)

with tab_table:
    display_df = df[["rank", "candidate_id", "score", "reasoning"]].copy()
    display_df["score"] = display_df["score"].apply(lambda s: f"{s:.2f}")
    display_df.columns = ["Rank", "Candidate ID", "Score", "Reasoning"]
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='color:#8b949e;font-size:0.78rem;text-align:center;'>"
    "HireIQ · Team: Ishika Mohod · Ashwini Koturwar · Dakshyani Borade · Redrob Hackathon"
    "</p>",
    unsafe_allow_html=True,
)