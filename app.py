#!/usr/bin/env python3
"""
ZenithRank — Recruiter Sandbox Dashboard
==========================================

A production-grade Streamlit application serving as the official hosted
Sandbox Demo for the ZenithRank talent intelligence platform.

Integrates with pipeline.anti_trap, pipeline.feature_engine, and
pipeline.reasoning_agent to process test batches of ≤100 candidate
profiles completely offline within the sandbox environment.

Usage:
    streamlit run app.py
"""

import csv
import io
import time
import ujson as json
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Modular pipeline dependencies ────────────────────────────────────────────
from pipeline.anti_trap import is_candidate_synthetic_trap, clean_text_lower
from pipeline.feature_engine import compute_profile_multipliers
from pipeline.reasoning_agent import build_candidate_justification


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

CORE_ARCHITECTURE_TOKENS = [
    "ranking", "retrieval", "search", "recommendation", "embeddings", "vector",
    "information retrieval", "learning to rank", "rerank", "learning-to-rank",
    "xgboost", "lightgbm", "faiss", "pinecone", "qdrant", "milvus", "weaviate",
    "elasticsearch", "opensearch", "ndcg", "mrr", "map", "hybrid search",
]

MAX_SANDBOX_CANDIDATES = 100


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ZenithRank | Recruiter Sandbox Dashboard",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Premium recruiter-grade styling
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── Google Font import ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global overrides ───────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
    }

    /* ── Remove default Streamlit padding ────────────────────────────── */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* ── Hero header ────────────────────────────────────────────────── */
    .hero-container {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.15) 0%, rgba(124, 58, 237, 0.10) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6, #06B6D4, #3B82F6);
        background-size: 300% 100%;
        animation: shimmer 3s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA, #A78BFA, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
        line-height: 1.6;
    }

    /* ── Metric cards ───────────────────────────────────────────────── */
    .metric-card {
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        text-align: center;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }
    .metric-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }
    .metric-scanned {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.12) 0%, rgba(59, 130, 246, 0.08) 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
    }
    .metric-scanned::after { background: linear-gradient(90deg, #3B82F6, #60A5FA); }
    .metric-trapped {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(248, 113, 113, 0.08) 100%);
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .metric-trapped::after { background: linear-gradient(90deg, #EF4444, #F87171); }
    .metric-matched {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(52, 211, 153, 0.08) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .metric-matched::after { background: linear-gradient(90deg, #10B981, #34D399); }

    .metric-icon { font-size: 2rem; margin-bottom: 0.3rem; }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.5rem;
    }
    .metric-scanned .metric-label { color: #60A5FA; }
    .metric-trapped .metric-label { color: #F87171; }
    .metric-matched .metric-label { color: #34D399; }

    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1;
    }
    .metric-scanned .metric-value { color: #93C5FD; }
    .metric-trapped .metric-value { color: #FCA5A5; }
    .metric-matched .metric-value { color: #6EE7B7; }

    /* ── Section headers ────────────────────────────────────────────── */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #E2E8F0;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Upload zone ────────────────────────────────────────────────── */
    .upload-zone {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 2px dashed rgba(99, 102, 241, 0.35);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.3s ease;
    }
    .upload-zone:hover {
        border-color: rgba(99, 102, 241, 0.6);
    }

    /* ── Token chip display ─────────────────────────────────────────── */
    .token-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 0.5rem;
    }
    .token-chip {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.10));
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #A5B4FC;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        font-family: 'Inter', monospace;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
    }
    .token-chip:hover {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.3), rgba(139, 92, 246, 0.2));
        transform: scale(1.05);
    }

    /* ── Info callout panel ──────────────────────────────────────────── */
    .info-callout {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.08), rgba(59, 130, 246, 0.06));
        border: 1px solid rgba(6, 182, 212, 0.25);
        border-radius: 12px;
        padding: 1rem 1.3rem;
        color: #67E8F9;
        font-size: 0.9rem;
        line-height: 1.5;
        margin: 1rem 0;
    }
    .info-callout strong { color: #A5F3FC; }

    /* ── Pipeline stage badges ──────────────────────────────────────── */
    .stage-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 3px 4px;
    }
    .stage-0 {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #FCA5A5;
    }
    .stage-1 {
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #93C5FD;
    }
    .stage-2 {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #6EE7B7;
    }

    /* ── Sidebar styling ────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        background: linear-gradient(135deg, #60A5FA, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.5rem;
    }

    /* ── Dataframe styling ──────────────────────────────────────────── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Download button styling ────────────────────────────────────── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #3B82F6, #8B5CF6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4) !important;
    }

    /* ── Primary button styling ─────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3B82F6, #6366F1) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
    }

    /* ── Expander styling ───────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.6) !important;
        border-radius: 10px !important;
        color: #CBD5E1 !important;
        font-weight: 500 !important;
    }

    /* ── Hide default streamlit footer ──────────────────────────────── */
    footer { visibility: hidden; }

    /* ── Divider line ───────────────────────────────────────────────── */
    .zen-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
        margin: 1.5rem 0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.markdown('<div class="zen-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### 📋 Active Evaluation Mandate")
    st.markdown("""
    <div class="info-callout">
        <strong>Role:</strong> Senior AI Engineer<br>
        <strong>Team:</strong> Founding Team — Redrob AI<br>
        <strong>Domain:</strong> Search & Retrieval at Scale<br>
        <strong>YOE Target:</strong> 5–9 years
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="zen-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### 🛡️ Pipeline Safeguards")
    st.markdown("""
        <span class="stage-badge stage-0">🚨 Stage 0 — Anti-Honeypot Shield</span><br>
        <span class="stage-badge stage-1">🔍 Stage 1 — Sparse Vector Engine</span><br>
        <span class="stage-badge stage-2">⚡ Stage 2 — Intent Multiplier Matrix</span>
    """, unsafe_allow_html=True)

    st.markdown('<div class="zen-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### ⚙️ Engine Parameters")
    st.markdown("""
    <div class="info-callout" style="font-size: 0.82rem;">
        <strong>TF-IDF Vocabulary:</strong> 23 domain tokens<br>
        <strong>N-gram Range:</strong> (1, 2)<br>
        <strong>Max Sandbox Batch:</strong> 100 profiles<br>
        <strong>Tie-Breaker:</strong> Ascending candidate_id<br>
        <strong>Output Schema:</strong> candidate_id, rank, score, reasoning
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="zen-divider"></div>', unsafe_allow_html=True)
    st.caption("ZenithRank Engine v1.0.0 · Offline CPU Sandboxed")


# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-container">
    <div class="hero-title">ZenithRank Talent Intelligence Sandbox</div>
    <div class="hero-subtitle">
        Stream candidate profile data feeds, isolate synthetic text loops and chronological honeypots,
        and output server-validated shortlists — all running offline on a single CPU core.
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN SPACE VIEWER
# ═══════════════════════════════════════════════════════════════════════════════

with st.expander("📐 View Target Semantic Verification Token Space", expanded=False):
    chips_html = "".join(
        f'<span class="token-chip">{token}</span>' for token in CORE_ARCHITECTURE_TOKENS
    )
    st.markdown(f'<div class="token-grid">{chips_html}</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-callout" style="margin-top: 1rem;">
        These <strong>23 domain tokens</strong> define the TF-IDF vocabulary space used to compute
        baseline semantic alignment between each candidate's profile text and the target role specification.
        Scores are then modulated by trajectory multipliers from the feature engine.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FILE UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="section-header">📤 Upload Candidate Sample Data Feed</div>',
    unsafe_allow_html=True,
)

st.markdown("""
<div class="upload-zone">
    <div style="text-align:center; color:#94A3B8; font-size:0.9rem;">
        Drop your profile snippet file below<br>
        <span style="color:#64748B; font-size:0.8rem;">
            Supports <code style="color:#A5B4FC;">sample_candidates.json</code> (array) or
            <code style="color:#A5B4FC;">candidates.jsonl</code> (line-delimited)
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Select candidate data file",
    type=["json", "jsonl"],
    label_visibility="collapsed",
)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if uploaded_file is not None:
    file_contents = uploaded_file.getvalue().decode("utf-8").strip()
    candidate_batch = []

    # ── Parse JSON array or JSONL lines ──────────────────────────────
    try:
        if file_contents.startswith("[") and file_contents.endswith("]"):
            candidate_batch = json.loads(file_contents)
        else:
            for line in file_contents.split("\n"):
                line = line.strip()
                if line:
                    candidate_batch.append(json.loads(line))
    except Exception as e:
        st.error(f"❌ **Structural Parsing Exception:** Encountered validation error mapping text lines: `{str(e)}`")
        st.stop()

    # ── Enforce sandbox batch limit ──────────────────────────────────
    total_records = len(candidate_batch)
    if total_records > MAX_SANDBOX_CANDIDATES:
        st.warning(
            f"⚠️ Sandbox limit: trimming input from **{total_records}** to "
            f"**{MAX_SANDBOX_CANDIDATES}** profiles for demo evaluation."
        )
        candidate_batch = candidate_batch[:MAX_SANDBOX_CANDIDATES]
        total_records = MAX_SANDBOX_CANDIDATES

    st.markdown(f"""
    <div class="info-callout">
        📦 <strong>Data stream active:</strong> Successfully cached
        <strong>{total_records}</strong> candidate profile records into the evaluation buffer.
    </div>
    """, unsafe_allow_html=True)

    # ── Execute button ───────────────────────────────────────────────
    if st.button("Execute ZenithRank Matching Logic", type="primary", use_container_width=True):
        t_start = time.perf_counter()

        # ── Setup TF-IDF vector space ────────────────────────────────
        vectorizer = TfidfVectorizer(
            vocabulary={token: i for i, token in enumerate(CORE_ARCHITECTURE_TOKENS)},
            ngram_range=(1, 2),
        )
        jd_vector_space = vectorizer.fit_transform([" ".join(CORE_ARCHITECTURE_TOKENS)])

        shortlist_buffer = []
        honeypots_defused = 0

        # ── Progress tracking ────────────────────────────────────────
        progress_bar = st.progress(0, text="Initialising pipeline stages...")
        status_container = st.empty()

        for idx, candidate in enumerate(candidate_batch):
            cid = candidate.get("candidate_id", "CAND_UNKNOWN")
            progress_pct = (idx + 1) / total_records
            progress_bar.progress(
                progress_pct,
                text=f"Analysing profile {idx + 1}/{total_records}  ·  `{cid}`",
            )

            # ── Stage 0: Anti-Trap Verification ──────────────────────
            if is_candidate_synthetic_trap(candidate):
                honeypots_defused += 1
                continue

            # ── Synthesize text corpus for TF-IDF ────────────────────
            profile = candidate.get("profile", {})
            skills = candidate.get("skills", [])
            history = candidate.get("career_history", [])

            summary_text = profile.get("summary", "")
            headline_text = profile.get("headline", "")
            title_text = profile.get("current_title", "")
            history_text = " ".join(job.get("description", "") for job in history)
            skill_text = " ".join(s.get("name", "") for s in skills)

            corpus_block = f"{summary_text} {headline_text} {title_text} {history_text} {skill_text}"
            cleaned_corpus = clean_text_lower(corpus_block)

            if not cleaned_corpus.strip():
                continue

            # ── Stage 1: Compute base cosine alignment ───────────────
            cand_matrix = vectorizer.transform([cleaned_corpus])
            base_cosine_score = float((cand_matrix * jd_vector_space.T).toarray()[0][0])

            if base_cosine_score > 0:
                # ── Stage 2: Apply trajectory multipliers ────────────
                multiplier = compute_profile_multipliers(candidate)
                final_score = round(base_cosine_score * multiplier, 6)

                shortlist_buffer.append({
                    "candidate_id": cid,
                    "score": final_score,
                    "_candidate_record": candidate,
                })

        elapsed = time.perf_counter() - t_start
        progress_bar.empty()
        status_container.empty()

        # ═════════════════════════════════════════════════════════════
        # RESULTS DISPLAY
        # ═════════════════════════════════════════════════════════════

        st.markdown(f"""
        <div class="info-callout" style="text-align:center;">
            🏁 Pipeline evaluation completed in <strong>{elapsed:.2f}s</strong>
        </div>
        """, unsafe_allow_html=True)

        # ── Metric cards ─────────────────────────────────────────────
        matches_found = len(shortlist_buffer)

        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            st.markdown(f"""
            <div class="metric-card metric-scanned">
                <div class="metric-icon">📬</div>
                <div class="metric-label">Total Ingested Profiles</div>
                <div class="metric-value">{total_records}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card metric-trapped">
                <div class="metric-icon">🚨</div>
                <div class="metric-label">Honeypots Caught & Dropped</div>
                <div class="metric-value">{honeypots_defused}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card metric-matched">
                <div class="metric-icon">🏆</div>
                <div class="metric-label">Qualified System Matches</div>
                <div class="metric-value">{matches_found}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="zen-divider"></div>', unsafe_allow_html=True)

        # ── Build ranked DataFrame ───────────────────────────────────
        if shortlist_buffer:
            results_df = pd.DataFrame(shortlist_buffer)

            # MANDATORY SORT: score descending, candidate_id ascending (tie-breaker)
            results_df = results_df.sort_values(
                by=["score", "candidate_id"],
                ascending=[False, True],
            ).reset_index(drop=True)

            # Slice to top 100
            shortlist_output = results_df.head(100).copy()
            shortlist_output["rank"] = range(1, len(shortlist_output) + 1)

            # Generate fact-anchored justifications via Stage 2
            shortlist_output["reasoning"] = shortlist_output.apply(
                lambda row: build_candidate_justification(
                    row["_candidate_record"], row["rank"], row["score"]
                ),
                axis=1,
            )

            # Final export columns in strict schema order
            display_cols = ["candidate_id", "rank", "score", "reasoning"]
            final_export_df = shortlist_output[display_cols].copy()

            # ── Shortlist table ──────────────────────────────────────
            st.markdown(
                '<div class="section-header">📋 System Generated Ranking Shortlist</div>',
                unsafe_allow_html=True,
            )

            st.dataframe(
                final_export_df,
                column_config={
                    "candidate_id": st.column_config.TextColumn(
                        "Candidate ID",
                        width="medium",
                    ),
                    "rank": st.column_config.NumberColumn(
                        "Rank",
                        format="%d",
                        width="small",
                    ),
                    "score": st.column_config.NumberColumn(
                        "Composite Score",
                        format="%.6f",
                        width="medium",
                    ),
                    "reasoning": st.column_config.TextColumn(
                        "Fact-Anchored Justification",
                        width="large",
                    ),
                },
                use_container_width=True,
                hide_index=True,
                height=min(len(final_export_df) * 40 + 60, 600),
            )

            # ── Score distribution chart ─────────────────────────────
            st.markdown(
                '<div class="section-header">📊 Score Distribution Across Shortlist</div>',
                unsafe_allow_html=True,
            )
            chart_df = final_export_df[["rank", "score"]].set_index("rank")
            st.area_chart(chart_df, color="#6366F1", use_container_width=True)

            # ── CSV Export ───────────────────────────────────────────
            st.markdown('<div class="zen-divider"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-header">💾 Export Server-Compliant Deliverable</div>',
                unsafe_allow_html=True,
            )

            csv_stream = io.StringIO()
            final_export_df.to_csv(csv_stream, index=False, quoting=csv.QUOTE_MINIMAL)
            csv_payload = csv_stream.getvalue()

            st.download_button(
                label="📥 Download Validated Submission CSV",
                data=csv_payload,
                file_name="sandbox_submission.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

            st.markdown("""
            <div class="info-callout" style="margin-top: 1rem; font-size: 0.82rem;">
                ✅ Output schema: <code>candidate_id, rank, score, reasoning</code><br>
                ✅ Scores monotonically non-increasing by rank<br>
                ✅ Ties broken by ascending alphabetical candidate_id<br>
                ✅ UTF-8 encoded with minimal quoting
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning(
                "⚠️ Zero candidate records cleared the initial matching threshold. "
                "Try uploading profiles with IR/ML/Search experience."
            )

else:
    # ── Empty state ──────────────────────────────────────────────────
    st.markdown("""
    <div class="info-callout" style="text-align: center; padding: 2rem;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">📂</div>
        <strong>Ready for evaluation</strong><br>
        Drop a <code>sample_candidates.json</code> array or <code>candidates.jsonl</code>
        file above to activate the pipeline and generate ranked shortlists.
    </div>
    """, unsafe_allow_html=True)

    # ── How it works section ─────────────────────────────────────────
    st.markdown(
        '<div class="section-header">⚡ How the Pipeline Works</div>',
        unsafe_allow_html=True,
    )

    stage_col1, stage_col2, stage_col3 = st.columns(3, gap="medium")
    with stage_col1:
        st.markdown("""
        <div class="metric-card metric-trapped" style="text-align:left;">
            <div style="font-size:1.3rem; margin-bottom:0.5rem;">🚨 Stage 0</div>
            <div style="color:#FCA5A5; font-weight:600; margin-bottom:0.4rem;">Anti-Honeypot Shield</div>
            <div style="color:#94A3B8; font-size:0.82rem; line-height:1.5;">
                Detects boilerplate text loops, chronological impossibilities
                (tenure > company age), and fake expert skills with 0 months duration.
                Removes all synthetic trap profiles before scoring.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with stage_col2:
        st.markdown("""
        <div class="metric-card metric-scanned" style="text-align:left;">
            <div style="font-size:1.3rem; margin-bottom:0.5rem;">🔍 Stage 1</div>
            <div style="color:#93C5FD; font-weight:600; margin-bottom:0.4rem;">Sparse Vector Engine</div>
            <div style="color:#94A3B8; font-size:0.82rem; line-height:1.5;">
                Computes TF-IDF cosine similarity against 23 domain tokens
                covering IR, search, ranking, vector databases, and evaluation
                metrics. Multiplied by trajectory and intent modifiers.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with stage_col3:
        st.markdown("""
        <div class="metric-card metric-matched" style="text-align:left;">
            <div style="font-size:1.3rem; margin-bottom:0.5rem;">⚡ Stage 2</div>
            <div style="color:#6EE7B7; font-weight:600; margin-bottom:0.4rem;">Intent Multiplier Matrix</div>
            <div style="color:#94A3B8; font-size:0.82rem; line-height:1.5;">
                Synthesizes 7 signal dimensions — experience band, corporate DNA,
                education tier, geography, recruiter response rate, notice period,
                and skill depth — into a composite trajectory multiplier.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="zen-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#475569; font-size:0.78rem; padding:1rem 0;">
    ZenithRank v1.0.0 · Built by IndianBisons (Aman Naurangabadi, Aniruddha Jadhav, Abdulkalam Qureshi, Akshay Patil)<br>
    <span style="color:#64748B;">Redrob AI — Best Candidate Discovery Challenge</span>
</div>
""", unsafe_allow_html=True)




