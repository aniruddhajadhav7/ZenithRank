#!/usr/bin/env python3
# """
# ZenithRank — Recruiter Intelligence Engine
# =========================================
# 
# A production-grade talent intelligence gateway designed to process high-throughput 
# candidate profile streams, evaluate core semantic alignment metrics, and generate 
# server-compliant verification shortlists completely offline.
# 
# Operational Significance:
#   - Automates semantic vector alignment using structured domain vocabularies.
#   - Mitigates profile spoofing and adversarial text loop manipulations in real time.
#   - Accelerates enterprise candidate shortlisting with rigorous sorting operations.
# """

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
"""
ZenithRank — Recruiter Intelligence Engine
=========================================


A production-grade talent intelligence gateway designed to process high-throughput 
candidate profile streams, evaluate core semantic alignment metrics, and generate 
server-compliant verification shortlists completely offline.


Operational Significance:
  - Automates semantic vector alignment using structured domain vocabularies.
  - Mitigates profile spoofing and adversarial text loop manipulations in real time.
  - Accelerates enterprise candidate shortlisting with rigorous sorting operations.
"""
st.set_page_config(
    page_title="ZenithRank | Intelligence Engine",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# THEME CONFIGURATION — Premium Black, Silver & Gold (Glassmorphic Reconstruct)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ====================================
       IMPORTS & GLOBAL SYSTEM OVERRIDES
    ==================================== */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* ====================================
       APP BACKGROUND (MATTE LUXURY VOID)
    ==================================== */
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(212, 175, 55, 0.08) 0%, transparent 35%),
            radial-gradient(circle at bottom right, rgba(255, 255, 255, 0.02) 0%, transparent 35%),
            linear-gradient(135deg, #050507 0%, #0A0A0F 40%, #050507 100%);
        color: white;
    }

    /* ====================================
       MAIN CONTAINER
    ==================================== */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* ====================================
       CORNER TEAM LOGO COMPONENT CONTAINER
    ==================================== */
    .corner-logo-container {
        position: absolute;
        top: 15px;
        right: 25px;
        display: flex;
        align-items: center;
        gap: 12px;
        background: rgba(10, 10, 15, 0.85);
        border: 1px solid rgba(212, 175, 55, 0.25);
        padding: 8px 18px;
        border-radius: 30px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        z-index: 999;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .corner-logo-text {
        font-family: 'Cinzel', serif;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 2px;
        background: linear-gradient(135deg, #FFFFFF 0%, #D4AF37 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .corner-logo-img-wrapper img {
        width: 22px;
        height: 22px;
        object-fit: contain;
        filter: drop-shadow(0px 0px 4px rgba(212, 175, 55, 0.6));
    }

    /* ====================================
       CINEMATIC GLASS HERO SECTION
    ==================================== */
    .bison-hero-canvas {
        position: relative;
        padding: 45px 50px;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.02), rgba(212, 175, 55, 0.02));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 30px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
    }
    
    .bison-hero-canvas::after {
        content: '𓃬';
        position: absolute;
        right: 5%;
        top: -15%;
        font-size: 13rem;
        color: rgba(212, 175, 55, 0.02);
        font-family: serif;
        pointer-events: none;
        transform: rotate(-12deg);
    }

    .hero-tagline {
        font-family: 'Cinzel', serif;
        font-size: 0.8rem;
        color: #D4AF37;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .hero-title-main {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #FFFFFF, #E2E8F0, #D4AF37);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }

    .hero-description {
        font-size: 1.02rem;
        color: #E2E8F0;
        max-width: 950px;
        line-height: 1.6;
        font-weight: 400;
        margin-bottom: 1.25rem;
    }

    .operational-significance-box {
        background: rgba(212, 175, 55, 0.03);
        border: 1px solid rgba(212, 175, 55, 0.15);
        border-radius: 12px;
        padding: 18px 22px;
        margin-top: 1rem;
    }

    .operational-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #D4AF37;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }

    .operational-list {
        margin: 0;
        padding-left: 20px;
        color: #94A3B8;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .operational-list li {
        margin-bottom: 0.35rem;
    }

    .operational-list li strong {
        color: #F8FAFC;
    }

    /* ====================================
       GLASS METRIC DISPLAY CARDS
    ==================================== */
    .stat-card {
        position: relative;
        padding: 30px;
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        overflow: hidden;
        transition: 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .stat-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 40px rgba(212, 175, 55, 0.1);
    }
    .stat-card::before {
        content: '';
        position: absolute;
        inset: 0;
        padding: 1px;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(212,175,55,0.2));
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
    }
    .stat-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .stat-value {
        font-size: 3.2rem;
        font-weight: 800;
        margin-top: 10px;
        line-height: 1;
    }
    /* Variant Text Highlights */
    .stat-gold .stat-value {
        background: linear-gradient(135deg, #FFFFFF 0%, #D4AF37 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-silver .stat-value {
        background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-dark .stat-value {
        color: #475569;
    }

    /* ====================================
       SECTION HEADERS
    ==================================== */
    .premium-header {
        font-family: 'Cinzel', serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFFFFF;
        letter-spacing: 2px;
        margin: 2.5rem 0 1.25rem 0;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .premium-header::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(212, 175, 55, 0.4), transparent);
    }

    /* ====================================
       GLASS FILE UPLOAD TARGET
    ==================================== */
    .premium-upload-box {
        padding: 40px;
        text-align: center;
        border-radius: 24px;
        border: 2px dashed rgba(212, 175, 55, 0.3);
        background: rgba(255, 255, 255, 0.02);
        transition: 0.3s ease;
        margin-bottom: 1rem;
    }
    .premium-upload-box:hover {
        border-color: #D4AF37;
        background: rgba(212, 175, 55, 0.04);
    }

    /* ====================================
       INTERACTIVE ACCESS CONTROL BUTTONS
    ==================================== */
    .stButton button {
        background: linear-gradient(135deg, #16161A, #0A0A0C) !important;
        color: #D4AF37 !important;
        border: 1px solid rgba(212, 175, 55, 0.4) !important;
        border-radius: 18px !important;
        font-weight: 700 !important;
        height: 54px !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        font-size: 0.85rem !important;
        transition: 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 100%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4) !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        border-color: #D4AF37 !important;
        box-shadow: 0 15px 40px rgba(212, 175, 55, 0.15) !important;
        color: #FFFFFF !important;
    }
    /* Execution Trigger Hierarchy */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #D4AF37, #AA8415) !important;
        color: #050507 !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 15px 40px rgba(212, 175, 55, 0.35) !important;
        color: #050507 !important;
    }

    /* ====================================
       SIDEBAR INTERFACE STRUCTURING
    ==================================== */
    section[data-testid="stSidebar"] {
        background: rgba(8, 8, 12, 0.85) !important;
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    section[data-testid="stSidebar"] .stMarkdown h4 {
        font-family: 'Cinzel', serif !important;
        color: #D4AF37 !important;
        font-size: 0.85rem !important;
        letter-spacing: 1.5px;
    }
    .sidebar-panel {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }
    .sidebar-row {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
    }
    .sidebar-row strong { color: #E2E8F0; }

    /* ====================================
       DATAFRAME GLASS OVERRIDES
    ==================================== */
    .stDataFrame {
        border-radius: 24px !important;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        background: rgba(10, 10, 14, 0.4);
    }

    /* ====================================
       TOKEN CHIPS & UTILITIES
    ==================================== */
    .token-flex-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 10px 0;
    }
    .luxury-chip {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #94A3B8;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.8rem;
        transition: 0.2s;
    }
    .luxury-chip:hover {
        border-color: #D4AF37;
        color: #FFFFFF;
    }
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 14px !important;
        color: #E2E8F0 !important;
    }
    .premium-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.06), transparent);
        margin: 3rem 0 1.5rem 0;
    }

    /* ====================================
       HIGH CONTRAST GLASS SCROLLBAR
    ==================================== */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-thumb { background: rgba(212, 175, 55, 0.4); border-radius: 20px; }
    ::-webkit-scrollbar-track { background: transparent; }

    /* Hide default platform rules */
    footer { visibility: hidden; }
</style>

<div class="corner-logo-container">
    <div class="corner-logo-img-wrapper">
        <img src="app/static/logo.png" onerror="this.src='logo.png';">
    </div>
    <div class="corner-logo-text">INDIAN BISONS</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION & META CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("<div style='padding: 1rem 0 0.5rem 0; text-align:center;'><h2 style='font-family:Cinzel, serif; color:#FFF; font-size:1.4rem; letter-spacing:2px;'>ZENITHRANK</h2></div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 1px; background: linear-gradient(90deg, transparent, rgba(212,175,55,0.3), transparent); margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

    st.markdown("#### 📋 TARGET MANDATE")
    st.markdown("""
    <div class="sidebar-panel">
        <div class="sidebar-row"><span>Role:</span><strong>Senior AI Engineer</strong></div>
        <div class="sidebar-row"><span>Team:</span><strong>Founding Team · Redrob</strong></div>
        <div class="sidebar-row"><span>Domain:</span><strong>Information Retrieval</strong></div>
        <div class="sidebar-row"><span>Target:</span><strong>5–9 Years YOE</strong></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🛡️ PIPELINE SAFEGUARDS")
    st.markdown("""
    <div class="sidebar-panel" style="gap: 8px; display: flex; flex-direction: column;">
        <div style="font-size:0.75rem; color:#FCA5A5; letter-spacing:0.5px;">● STAGE 0 — ANTI-HONEYPOT SHIELD</div>
        <div style="font-size:0.75rem; color:#93C5FD; letter-spacing:0.5px;">● STAGE 1 — SPARSE VECTOR MATRIX</div>
        <div style="font-size:0.75rem; color:#6EE7B7; letter-spacing:0.5px;">● STAGE 2 — INTENT MULTIPLIER</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### ⚙️ ENGINE PARAMETERS")
    st.markdown("""
    <div class="sidebar-panel" style="font-size: 0.8rem; color:#64748B;">
        <div class="sidebar-row"><span>Vocabulary:</span><span>23 Domain Tokens</span></div>
        <div class="sidebar-row"><span>N-gram Range:</span><span>(1, 2)</span></div>
        <div class="sidebar-row"><span>Batch Size Limit:</span><span>100 Profiles</span></div>
        <div class="sidebar-row"><span>Tie-Breaker:</span><span>candidate_id (Asc)</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Engine Core v1.0.0 · Offline Operational Environment")


# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER (CLEAN OPERATIONAL DESCRIPTION OUTPUT)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="bison-hero-canvas">
    <div class="hero-tagline">Indian Bisons Deployment</div>
    <div class="hero-title-main">ZenithRank Recruiter Intelligence Engine</div>
    <div class="hero-description">
        A production-grade talent intelligence gateway designed to process high-throughput 
        candidate profile streams, evaluate core semantic alignment metrics, and generate 
        server-compliant verification shortlists completely offline.
    </div>
    <div class="operational-significance-box">
        <div class="operational-title">Operational Significance</div>
        <ul class="operational-list">
            <li><strong>Semantic Alignment Automation:</strong> Maps continuous target applicant tracking streams against highly structured, multi-dimensional domain token dictionaries.</li>
            <li><strong>Real-time Loop Mitigation:</strong> Detects and neutralizes anomalous profile spoofing attempts, synthetic data constructs, and adversarial text loop manipulations instantly.</li>
            <li><strong>Enterprise Shortlist Acceleration:</strong> Runs high-dimension structural vector sorting algorithms to provide rigorous candidate discovery transparency.</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPANDABLE TOKEN ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

with st.expander("📐 VIEW TARGET SEMANTIC VERIFICATION MATRIX", expanded=False):
    chips_html = "".join(f'<span class="luxury-chip">{token}</span>' for token in CORE_ARCHITECTURE_TOKENS)
    st.markdown(f'<div class="token-flex-container">{chips_html}</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color: #64748B; font-size: 0.85rem; margin-top: 0.5rem; line-height: 1.5;">
        These 23 core domain tokens form the fixed dictionary space for calculating the structural alignment coefficient. 
        Raw TF-IDF score targets are subsequently modulated via dynamic trajectory vector weights.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA INGESTION WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="premium-header">📤 DATASTREAM INGESTION ROUTINE</div>', unsafe_allow_html=True)

st.markdown("""
<div class="premium-upload-box">
    <div style="color: #E2E8F0; font-size: 0.9rem; font-weight: 500; margin-bottom: 0.25rem;">
        Mount Candidate Target Payload File
    </div>
    <div style="color: #64748B; font-size: 0.8rem;">
        Accepts structured arrays (<code style="color: #D4AF37;">sample_candidates.json</code>) or flat streams (<code style="color: #D4AF37;">candidates.jsonl</code>)
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Select candidate data file",
    type=["json", "jsonl"],
    label_visibility="collapsed",
)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPILATION & DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

if uploaded_file is not None:
    file_contents = uploaded_file.getvalue().decode("utf-8").strip()
    candidate_batch = []

    try:
        if file_contents.startswith("[") and file_contents.endswith("]"):
            candidate_batch = json.loads(file_contents)
        else:
            for line in file_contents.split("\n"):
                line = line.strip()
                if line:
                    candidate_batch.append(json.loads(line))
    except Exception as e:
        st.error(f"❌ **Structural Parsing Exception:** Engine failure mapped payload configuration: `{str(e)}`")
        st.stop()

    total_records = len(candidate_batch)
    if total_records > MAX_SANDBOX_CANDIDATES:
        st.warning(f"⚠️ Limit constraint exceeded: trimming incoming feed from {total_records} down to top {MAX_SANDBOX_CANDIDATES} profiles.")
        candidate_batch = candidate_batch[:MAX_SANDBOX_CANDIDATES]
        total_records = MAX_SANDBOX_CANDIDATES

    # Trigger Evaluation Button
    if st.button("EXECUTE PARALLEL MATCHING ENGINE", type="primary", use_container_width=True):
        t_start = time.perf_counter()

        # Vectorization Setup
        vectorizer = TfidfVectorizer(
            vocabulary={token: i for i, token in enumerate(CORE_ARCHITECTURE_TOKENS)},
            ngram_range=(1, 2),
        )
        jd_vector_space = vectorizer.fit_transform([" ".join(CORE_ARCHITECTURE_TOKENS)])

        shortlist_buffer = []
        honeypots_defused = 0

        progress_bar = st.progress(0, text="Staging processing nodes...")

        for idx, candidate in enumerate(candidate_batch):
            cid = candidate.get("candidate_id", "CAND_UNKNOWN")
            progress_bar.progress((idx + 1) / total_records, text=f"Evaluating Vector Arrays {idx + 1}/{total_records} · File Pointer: `{cid}`")

            # Stage 0: Verification
            if is_candidate_synthetic_trap(candidate):
                honeypots_defused += 1
                continue

            # Context Text Block Assembly
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

            # Stage 1: Similarity Core Mapping
            cand_matrix = vectorizer.transform([cleaned_corpus])
            base_cosine_score = float((cand_matrix * jd_vector_space.T).toarray()[0][0])

            if base_cosine_score > 0:
                # Stage 2: Intent Multipliers Matrix
                multiplier = compute_profile_multipliers(candidate)
                final_score = round(base_cosine_score * multiplier, 6)

                shortlist_buffer.append({
                    "candidate_id": cid,
                    "score": final_score,
                    "_candidate_record": candidate,
                })

        elapsed = time.perf_counter() - t_start
        progress_bar.empty()

        # ── STATISTICAL LUXURY GRID ───────────────────────────────────
        matches_found = len(shortlist_buffer)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class="stat-card stat-silver">
                <div class="stat-label">INGESTED FEEDS</div>
                <div class="stat-value">{total_records}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="stat-card stat-dark">
                <div class="stat-label">DEFUSED TRAPS</div>
                <div class="stat-value">{honeypots_defused}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="stat-card stat-gold">
                <div class="stat-label">MATCHES ARCHITECTED</div>
                <div class="stat-value">{matches_found}</div>
            </div>
            """, unsafe_allow_html=True)

        # Processing Context Output
        st.markdown(f"<div style='text-align:right; color:#64748B; font-size:0.8rem; margin-top:0.5rem;'>Pipeline computed completely in {elapsed:.4f} seconds</div>", unsafe_allow_html=True)

        # Data Frame Formulation
        if shortlist_buffer:
            results_df = pd.DataFrame(shortlist_buffer)
            results_df = results_df.sort_values(by=["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)

            shortlist_output = results_df.head(100).copy()
            shortlist_output["rank"] = range(1, len(shortlist_output) + 1)
            shortlist_output["reasoning"] = shortlist_output.apply(
                lambda row: build_candidate_justification(row["_candidate_record"], row["rank"], row["score"]), axis=1
            )

            display_cols = ["candidate_id", "rank", "score", "reasoning"]
            final_export_df = shortlist_output[display_cols].copy()

            # Result Shortlist Table
            st.markdown('<div class="premium-header">📋 MACHINE-GENERATED EVALUATION RESULTS</div>', unsafe_allow_html=True)
            st.dataframe(
                final_export_df,
                column_config={
                    "candidate_id": st.column_config.TextColumn("CANDIDATE HASH", width="medium"),
                    "rank": st.column_config.NumberColumn("RANK", format="%d", width="small"),
                    "score": st.column_config.NumberColumn("COMPOSITE SCORE", format="%.6f", width="medium"),
                    "reasoning": st.column_config.TextColumn("FACT-ANCHORED EVIDENCE REPORT", width="large"),
                },
                use_container_width=True,
                hide_index=True,
                height=min(len(final_export_df) * 40 + 60, 500),
            )

            # High contrast analytics breakdown
            st.markdown('<div class="premium-header">📊 SIGNAL DECAY MATRIX</div>', unsafe_allow_html=True)
            chart_df = final_export_df[["rank", "score"]].set_index("rank")
            st.area_chart(chart_df, color="#D4AF37", use_container_width=True)

            # Export Trigger Area
            st.markdown('<div class="premium-header">💾 DOWNSTREAM METADATA EXPORT</div>', unsafe_allow_html=True)
            csv_stream = io.StringIO()
            final_export_df.to_csv(csv_stream, index=False, quoting=csv.QUOTE_MINIMAL)
            
            st.download_button(
                label="EXPORT COMPLIANT PIPELINE DELIVERABLE (CSV)",
                data=csv_stream.getvalue(),
                file_name="zenithrank_bison_shortlist.csv",
                mime="text/csv",
                use_container_width=True,
            )

        else:
            st.warning("No candidate profiles matched the specified architectural vectors.")

else:
    # Empty State Interface Component
    st.markdown("""
    <div style="background: rgba(255,255,255,0.01); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.05); border-radius: 24px; padding: 4rem 2rem; text-align: center; margin-top:2rem;">
        <div style="font-size: 2.5rem; margin-bottom: 1rem; filter: grayscale(100%);">𓃬</div>
        <div style="font-family: 'Cinzel', serif; font-size: 1rem; color: #D4AF37; letter-spacing: 2px; margin-bottom: 0.25rem;">
            AWAITING INPUT TRANSACTION DATA
        </div>
        <div style="color: #64748B; font-size: 0.85rem; max-width: 500px; margin: 0 auto; line-height:1.5;">
            Mount candidate vector streams to fire target execution logic arrays and run deep scoring pipelines.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER CREDITS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#475569; font-size:0.75rem; font-weight:400; letter-spacing: 1px;">
    ZENITHRANK v1.0.0 · POWERED BY <span style="color:#D4AF37; font-family:'Cinzel', serif; font-weight:700;">INDIAN BISONS</span><br>
    <span style="color:#334155; font-size:0.7rem;">AMAN NAURANGABADI · ANIRUDDHA JADHAV · ABDULKALAM QURESHI · AKSHAY PATIL</span>
</div>
""", unsafe_allow_html=True)