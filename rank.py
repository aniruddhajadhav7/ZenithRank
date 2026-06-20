#!/usr/bin/env python3
"""
ZenithRank — Offline Multi-Stage Candidate Ranking Engine
==========================================================

Processes a pool of up to 100,000 candidate profiles from a JSONL(.gz) file,
applies a cascade of anti-trap filters, TF-IDF sparse retrieval, and
programmatic trajectory multipliers to produce a deterministic top-100
shortlist.

Performance targets:
  • Wall clock:  < 5 minutes on a single CPU core
  • Peak RAM:    < 16 GB (typically ~800 MB for 100K profiles)
  • Honeypots:   0% in final shortlist

Usage:
  python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv

Output:
  A UTF-8 CSV with exactly 101 lines (1 header + 100 data rows).
  Columns: candidate_id, rank, score, reasoning
  Scores are monotonically non-increasing by rank.
  Ties are broken by ascending alphabetical candidate_id.
"""

import argparse
import csv
import gzip
import io
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ujson

# ── scikit-learn sparse retrieval ────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Local pipeline stages ───────────────────────────────────────────────────
from pipeline.anti_trap import clean_text_lower, is_candidate_synthetic_trap
from pipeline.feature_engine import compute_profile_multipliers
from pipeline.reasoning_agent import build_candidate_justification


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

TOP_K = 100
TFIDF_MAX_FEATURES = 25_000
TFIDF_NGRAM_RANGE = (1, 2)

# The ideal-candidate query document used for TF-IDF cosine similarity.
# This is engineered to surface profiles with production IR/ML/Search depth.
IDEAL_CANDIDATE_QUERY = """
information retrieval search engine ranking system dense retrieval hybrid retrieval
vector search approximate nearest neighbor faiss elasticsearch solr lucene
recommendation engine recommender system collaborative filtering content-based
learning to rank NDCG MRR MAP evaluation metrics A/B testing online evaluation
click model relevance feedback query understanding query rewriting
machine learning deep learning natural language processing NLP
transformer BERT GPT encoder decoder attention mechanism
PyTorch TensorFlow scikit-learn XGBoost LightGBM feature engineering
embedding sentence transformer bi-encoder cross-encoder
reranking candidate generation two-tower model
production deployment microservice API backend infrastructure
low latency high throughput distributed system pipeline
MLOps model serving inference optimization quantization
data pipeline ETL feature store experiment tracking
Python Java Scala Go distributed computing Spark
search relevance semantic search lexical search BM25 TF-IDF
personalization user modeling session modeling
knowledge graph entity linking named entity recognition
document understanding passage retrieval open-domain QA
"""

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ZenithRank")


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_candidates(filepath: str) -> List[Dict[str, Any]]:
    """
    Load candidate records from a JSONL or JSONL.GZ file.
    Each line is a complete JSON object representing one candidate.
    """
    candidates: List[Dict[str, Any]] = []

    opener = gzip.open if filepath.endswith(".gz") else open
    mode = "rt" if filepath.endswith(".gz") else "r"

    log.info("Loading candidates from: %s", filepath)
    with opener(filepath, mode, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = ujson.loads(line)
                candidates.append(record)
            except (ujson.JSONDecodeError, ValueError) as exc:
                log.warning("Skipping malformed line %d: %s", line_num, exc)

    log.info("Loaded %d raw candidate records.", len(candidates))
    return candidates


# ═══════════════════════════════════════════════════════════════════════════
# TEXT DOCUMENT CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def build_candidate_document(candidate: Dict[str, Any]) -> str:
    """
    Flatten a candidate record into a single text document suitable for
    TF-IDF vectorisation.  Concatenates summary, headline, title, skills,
    career history titles/descriptions, and education fields.
    """
    parts: List[str] = []

    profile = candidate.get("profile", {})
    parts.append(clean_text_lower(profile.get("summary", "")))
    parts.append(clean_text_lower(profile.get("headline", "")))
    parts.append(clean_text_lower(profile.get("current_title", "")))
    parts.append(clean_text_lower(profile.get("current_company", "")))

    # Skills
    for skill in candidate.get("skills", []):
        name = clean_text_lower(skill.get("name", ""))
        if name:
            parts.append(name)
            # Repeat skill names proportional to proficiency for soft boosting
            proficiency = clean_text_lower(skill.get("proficiency", ""))
            if proficiency == "expert":
                parts.append(name)
                parts.append(name)
            elif proficiency == "advanced":
                parts.append(name)

    # Career history
    for job in candidate.get("career_history", []):
        parts.append(clean_text_lower(job.get("title", "")))
        parts.append(clean_text_lower(job.get("company", "")))
        parts.append(clean_text_lower(job.get("description", "")))

    # Education
    for edu in candidate.get("education", []):
        parts.append(clean_text_lower(edu.get("degree", "")))
        parts.append(clean_text_lower(edu.get("field_of_study", "")))
        parts.append(clean_text_lower(edu.get("institution", "")))

    # Certifications / projects
    for cert in candidate.get("certifications", []):
        parts.append(clean_text_lower(cert.get("name", "")))

    for proj in candidate.get("projects", []):
        parts.append(clean_text_lower(proj.get("title", "")))
        parts.append(clean_text_lower(proj.get("description", "")))

    return " ".join(p for p in parts if p)


# ═══════════════════════════════════════════════════════════════════════════
# TF-IDF RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════

def compute_tfidf_scores(
    documents: List[str],
    query: str,
) -> np.ndarray:
    """
    Fit a TF-IDF vectoriser on the candidate documents, transform the query,
    and return cosine similarity scores as a 1-D numpy array.
    """
    log.info(
        "Building TF-IDF matrix (max_features=%d, ngrams=%s)...",
        TFIDF_MAX_FEATURES,
        TFIDF_NGRAM_RANGE,
    )

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
        stop_words="english",
        dtype=np.float32,
    )

    # Fit on candidates + query together to share vocabulary
    all_docs = documents + [query]
    tfidf_matrix = vectorizer.fit_transform(all_docs)

    # Query vector is the last row
    query_vec = tfidf_matrix[-1:]
    candidate_matrix = tfidf_matrix[:-1]

    log.info(
        "TF-IDF matrix shape: %s  |  Vocabulary size: %d",
        candidate_matrix.shape,
        len(vectorizer.vocabulary_),
    )

    scores = cosine_similarity(query_vec, candidate_matrix).flatten()
    return scores


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORING & RANKING
# ═══════════════════════════════════════════════════════════════════════════

def rank_candidates(
    candidates: List[Dict[str, Any]],
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Full ranking pipeline:
      1. Anti-trap filter
      2. Build text documents
      3. TF-IDF retrieval scores
      4. Multiply by profile multipliers
      5. Sort: descending score → ascending candidate_id (tie-breaker)
      6. Return top-K with (candidate_id, composite_score, candidate_record)
    """
    t0 = time.perf_counter()

    # ── Stage 0: Anti-trap filtering ─────────────────────────────────
    log.info("Stage 0: Running anti-trap filter on %d candidates...", len(candidates))
    clean_candidates: List[Dict[str, Any]] = []
    trap_count = 0
    for cand in candidates:
        if is_candidate_synthetic_trap(cand):
            trap_count += 1
        else:
            clean_candidates.append(cand)
    log.info(
        "Anti-trap: removed %d synthetic/honeypot profiles. %d remain.",
        trap_count,
        len(clean_candidates),
    )

    if len(clean_candidates) == 0:
        log.error("No candidates survived the anti-trap filter!")
        return []

    # ── Stage 1: Document construction ───────────────────────────────
    log.info("Stage 1: Building candidate text documents...")
    documents: List[str] = []
    candidate_ids: List[str] = []
    valid_candidates: List[Dict[str, Any]] = []

    for cand in clean_candidates:
        cid = cand.get("candidate_id", "")
        if not cid:
            # Also check nested profile for candidate_id
            cid = cand.get("profile", {}).get("candidate_id", "")
        if not cid:
            continue  # skip records without ID

        doc = build_candidate_document(cand)
        if not doc.strip():
            continue

        documents.append(doc)
        candidate_ids.append(str(cid))
        valid_candidates.append(cand)

    log.info("Built %d text documents for TF-IDF.", len(documents))

    # ── Stage 2: TF-IDF retrieval ────────────────────────────────────
    tfidf_scores = compute_tfidf_scores(documents, IDEAL_CANDIDATE_QUERY)

    # ── Stage 3: Profile multiplier synthesis ────────────────────────
    log.info("Stage 3: Computing profile multipliers...")
    multipliers = np.array(
        [compute_profile_multipliers(cand) for cand in valid_candidates],
        dtype=np.float64,
    )

    # ── Composite score = TF-IDF similarity × multiplier ─────────────
    composite_scores = tfidf_scores.astype(np.float64) * multipliers

    # Clamp negative scores to zero
    composite_scores = np.maximum(composite_scores, 0.0)

    elapsed_scoring = time.perf_counter() - t0
    log.info("Scoring complete in %.2f seconds.", elapsed_scoring)

    # ── Stage 4: Deterministic sorting ───────────────────────────────
    # Primary: descending score.  Tie-breaker: ascending candidate_id.
    scored_tuples: List[Tuple[str, float, Dict[str, Any]]] = [
        (candidate_ids[i], composite_scores[i], valid_candidates[i])
        for i in range(len(valid_candidates))
    ]

    # Sort by (-score, candidate_id) to satisfy the validator's tie-breaker
    scored_tuples.sort(key=lambda t: (-t[1], t[0]))

    return scored_tuples[:TOP_K]


# ═══════════════════════════════════════════════════════════════════════════
# CSV OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def write_submission_csv(
    ranked: List[Tuple[str, float, Dict[str, Any]]],
    output_path: str,
) -> None:
    """
    Write the final submission CSV.

    Schema: candidate_id,rank,score,reasoning
    - Exactly 100 data rows + 1 header.
    - Scores monotonically non-increasing.
    - UTF-8 encoding.
    """
    log.info("Writing submission CSV to: %s", output_path)

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank_idx, (cid, score, candidate) in enumerate(ranked, start=1):
            reasoning = build_candidate_justification(candidate, rank_idx, score)
            # Sanitise reasoning: remove newlines, ensure no unquoted commas break CSV
            reasoning = reasoning.replace("\n", " ").replace("\r", " ").strip()
            writer.writerow([cid, rank_idx, f"{score:.6f}", reasoning])

    # ── Post-write validation ────────────────────────────────────────
    _validate_output(output_path)


def _validate_output(filepath: str) -> None:
    """
    Self-check: verify the output CSV satisfies all structural constraints
    before the user submits it.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)

    header = rows[0]
    data_rows = rows[1:]

    # Check header
    expected_header = ["candidate_id", "rank", "score", "reasoning"]
    assert header == expected_header, (
        f"Header mismatch: expected {expected_header}, got {header}"
    )

    # Check row count
    assert len(data_rows) == TOP_K, (
        f"Expected exactly {TOP_K} data rows, got {len(data_rows)}"
    )

    # Check monotonically non-increasing scores
    prev_score = float("inf")
    for i, row in enumerate(data_rows):
        rank = int(row[1])
        score = float(row[2])
        assert rank == i + 1, f"Rank mismatch at row {i+1}: expected {i+1}, got {rank}"
        assert score <= prev_score + 1e-9, (
            f"Score not monotonically non-increasing at rank {rank}: "
            f"{score} > {prev_score}"
        )
        prev_score = score

    # Check tie-breaker ordering
    for i in range(len(data_rows) - 1):
        score_a = float(data_rows[i][2])
        score_b = float(data_rows[i + 1][2])
        if abs(score_a - score_b) < 1e-9:  # effectively equal
            cid_a = data_rows[i][0]
            cid_b = data_rows[i + 1][0]
            assert cid_a <= cid_b, (
                f"Tie-breaker violation: '{cid_a}' should come before '{cid_b}' "
                f"(ascending alphabetical) at ranks {i+1}/{i+2}"
            )

    log.info("✓ Output validation passed: %d rows, correct schema, monotonic scores.", len(data_rows))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ZenithRank",
        description="Offline multi-stage candidate ranking engine for the Redrob Challenge.",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="Path to the candidates JSONL or JSONL.GZ file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./submission.csv",
        help="Output path for the submission CSV (default: ./submission.csv).",
    )
    args = parser.parse_args()

    # ── Validate input ───────────────────────────────────────────────
    if not os.path.isfile(args.candidates):
        log.error("Candidates file not found: %s", args.candidates)
        sys.exit(1)

    # ── Run pipeline ─────────────────────────────────────────────────
    t_start = time.perf_counter()
    log.info("=" * 68)
    log.info("  ZenithRank — Candidate Discovery Engine")
    log.info("=" * 68)

    candidates = load_candidates(args.candidates)

    if len(candidates) == 0:
        log.error("No candidates loaded. Check input file format.")
        sys.exit(1)

    ranked = rank_candidates(candidates)

    if len(ranked) < TOP_K:
        log.warning(
            "Only %d candidates survived filtering (need %d). "
            "Output will contain fewer rows than required.",
            len(ranked),
            TOP_K,
        )

    write_submission_csv(ranked, args.out)

    t_total = time.perf_counter() - t_start
    log.info("=" * 68)
    log.info("  Pipeline complete in %.2f seconds.", t_total)
    log.info("  Output: %s", os.path.abspath(args.out))
    log.info("=" * 68)


if __name__ == "__main__":
    main()
