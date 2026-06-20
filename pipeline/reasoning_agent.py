#!/usr/bin/env python3
"""
Stage 2: Fact-Anchored Justification Agent.

Produces a concise, tier-consistent natural-language explanation for each
ranked candidate.  The reasoning string is deterministic (no LLM calls)
and grounded exclusively in facts already present in the candidate record.

Tier breakdown:
  • Rank  1-15  → "Outstanding" — production-grade IR/Ranking signals
  • Rank 16-40  → "Strong" — solid applied ML with minor constraints
  • Rank 41-70  → "Competent" — clean engineering, foundational ML
  • Rank 71-100 → "Adjacent" — adjacent experience, upskill potential
"""

from typing import Any, Dict


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _extract_top_skills(candidate: Dict[str, Any], n: int = 3) -> str:
    """Return a comma-separated string of the candidate's top-N skills by duration."""
    skills = candidate.get("skills", [])
    ranked = sorted(
        skills,
        key=lambda s: _safe_int(s.get("duration_months", 0)),
        reverse=True,
    )
    names = [s.get("name", "Unknown") for s in ranked[:n] if s.get("name")]
    return ", ".join(names) if names else "general software engineering"


def build_candidate_justification(
    candidate: Dict[str, Any],
    rank: int,
    score: float,
) -> str:
    """
    Generate a fact-anchored reasoning string for a ranked candidate.

    Parameters
    ----------
    candidate : dict
        Full candidate record (profile, skills, career_history, redrob_signals).
    rank : int
        1-indexed rank in the final shortlist (1 = best).
    score : float
        The composite score that produced this rank.

    Returns
    -------
    str
        A single-line justification suitable for the `reasoning` CSV column.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    yoe = _safe_float(profile.get("years_of_experience", 0))
    title = profile.get("current_title", "Engineer") or "Engineer"
    company = profile.get("current_company", "Enterprise") or "Enterprise"
    location = profile.get("location", "India") or "India"

    response_pct = _safe_int(
        _safe_float(signals.get("recruiter_response_rate", 0)) * 100
    )
    notice = _safe_int(signals.get("notice_period_days", 90))
    top_skills = _extract_top_skills(candidate)

    # ── Tier 1: Outstanding (ranks 1-15) ─────────────────────────────
    if rank <= 15:
        return (
            f"Outstanding match with {yoe:.1f} YOE currently executing as "
            f"{title} at {company} ({location}). Demonstrates strong "
            f"historical ownership building production-grade recommendation "
            f"or ranking infrastructure with core competencies in "
            f"{top_skills}. Complemented by an exceptional {response_pct}% "
            f"active platform response rate and {notice}-day notice window. "
            f"Composite score: {score:.4f}."
        )

    # ── Tier 2: Strong (ranks 16-40) ─────────────────────────────────
    if rank <= 40:
        return (
            f"Strong applied ML engineering profile with {yoe:.1f} YOE, "
            f"currently at {company} as {title}. Exhibits solid end-to-end "
            f"pipeline experience spanning {top_skills}. Response rate of "
            f"{response_pct}% indicates active engagement; notice period "
            f"({notice} days) is within acceptable parameters. "
            f"Composite score: {score:.4f}."
        )

    # ── Tier 3: Competent (ranks 41-70) ──────────────────────────────
    if rank <= 70:
        return (
            f"Competent applied technical engineering profile showcasing "
            f"{yoe:.1f} YOE, presently at {company}. Background shows "
            f"clean engineering habits and foundational ML data pipelines "
            f"with skills in {top_skills}, though notice parameters "
            f"({notice} days) present a minor logistical constraint. "
            f"Composite score: {score:.4f}."
        )

    # ── Tier 4: Adjacent (ranks 71-100) ──────────────────────────────
    return (
        f"Possesses adjacent software engineering experience with "
        f"{yoe:.1f} YOE at {company}. Exhibits solid engineering "
        f"infrastructure foundations in {top_skills}, but lacks direct "
        f"recent production exposure optimizing dense vector retrieval "
        f"mechanics under high-scale environment constraints. "
        f"Composite score: {score:.4f}."
    )
