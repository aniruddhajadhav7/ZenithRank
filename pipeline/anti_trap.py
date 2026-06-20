#!/usr/bin/env python3
"""
Stage 0: Anti-Trap Shield.
Detects and drops automated honeypots, Section-7 chronological impossibilities,
and synthetic boilerplate text loops before any scoring occurs.

Traps addressed:
  1. Summary Boilerplate Trap — non-tech profiles padded with AI skill cards.
  2. Section 7 Chronological Honeypots — impossible tenure/company-age combos
     and "expert" skills with exactly 0 months of use.
  3. Mismatched persona injection — Operations Managers, Accountants, etc.
     wearing ML/AI keyword costumes.
"""

import re
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def clean_text_lower(text: Any) -> str:
    """Normalise any value to a lowercase stripped string."""
    if text is None:
        return ""
    return str(text).lower().strip()


# ---------------------------------------------------------------------------
# Boilerplate fingerprints
# ---------------------------------------------------------------------------

# Exact sub-phrases recycled across injected synthetic summaries
_BOILERPLATE_FRAGMENTS = [
    "my professional background is in marketing manager",
    "my professional background is in operations manager",
    "my professional background is in accountant",
    "my professional background is in civil engineer",
    "my professional background is in hr manager",
    "i've built and led teams",
]

# Titles that are structurally incompatible with the IR/ML target role
_HARD_REJECT_TITLES = frozenset([
    "marketing manager",
    "civil engineer",
    "accountant",
    "hr manager",
    "operations manager",
    "customer support",
    "customer support manager",
    "sales manager",
    "business development manager",
    "content writer",
    "graphic designer",
    "financial analyst",
    "general manager",
    "administrative assistant",
    "office manager",
    "receptionist",
    "social media manager",
    "event manager",
    "supply chain manager",
    "logistics manager",
    "procurement manager",
    "legal counsel",
    "compliance officer",
])

# Minimal tech-title tokens that override a suspicious summary
_TECH_ROLE_TOKENS = frozenset([
    "ml", "machine learning", "ai", "artificial intelligence",
    "search", "nlp", "natural language", "engineer", "developer",
    "data scientist", "data engineer", "sde", "swe", "software",
    "retrieval", "ranking", "recommendation", "deep learning",
    "research scientist", "applied scientist",
])


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------

def _has_boilerplate_summary(summary: str) -> bool:
    """Return True if the summary contains a known synthetic text loop."""
    for fragment in _BOILERPLATE_FRAGMENTS:
        if fragment in summary:
            return True
    return False


def _title_or_headline_is_tech(current_title: str, headline: str) -> bool:
    """Return True if either the title or headline contains a tech-role token."""
    combined = f"{current_title} {headline}"
    for token in _TECH_ROLE_TOKENS:
        if token in combined:
            return True
    return False


def _has_impossible_skill_claims(skills: list) -> bool:
    """
    Flag candidates who list ≥ 3 skills at "expert" proficiency with exactly
    0 months of usage — a hallmark of Section 7 synthetic injection.
    """
    fraudulent_count = 0
    for skill in skills:
        proficiency = clean_text_lower(skill.get("proficiency", ""))
        duration = 0
        try:
            duration = int(skill.get("duration_months", 0))
        except (ValueError, TypeError):
            duration = 0
        if proficiency == "expert" and duration == 0:
            fraudulent_count += 1
        if fraudulent_count >= 3:
            return True
    return False


def _has_chronological_impossibilities(career_history: list) -> bool:
    """
    Detect tenure-exceeds-company-age traps.
    If a candidate claims N years at a company that has only existed for
    fewer than N years, the profile is structurally impossible.
    """
    for job in career_history:
        company_age_years = None
        try:
            company_age_years = float(job.get("company_age_years", None) or 999)
        except (ValueError, TypeError):
            continue  # missing data ≠ fraud

        duration_months = 0
        try:
            duration_months = float(job.get("duration_months", 0))
        except (ValueError, TypeError):
            continue

        tenure_years = duration_months / 12.0
        if company_age_years < 900 and tenure_years > company_age_years:
            return True
    return False


def _has_zero_duration_all_skills(skills: list) -> bool:
    """
    Catch profiles where *every* skill has 0 months — a different variant
    of the honeypot pattern where no real work history backs the skills.
    """
    if not skills:
        return False
    for skill in skills:
        try:
            if int(skill.get("duration_months", 0)) > 0:
                return False
        except (ValueError, TypeError):
            pass
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_candidate_synthetic_trap(candidate: Dict[str, Any]) -> bool:
    """
    Master gate: return True if the candidate profile exhibits one or more
    structural markers of a synthetic / honeypot injection.

    This function is intentionally *aggressive* — false positives are
    acceptable because they only exclude profiles that were never real
    candidates in the first place.
    """
    profile = candidate.get("profile", {})
    summary = clean_text_lower(profile.get("summary", ""))
    current_title = clean_text_lower(profile.get("current_title", ""))
    headline = clean_text_lower(profile.get("headline", ""))
    skills = candidate.get("skills", [])
    career_history = candidate.get("career_history", [])

    # ── Trap 1: Boilerplate summary with non-tech title ──────────────
    if _has_boilerplate_summary(summary):
        if not _title_or_headline_is_tech(current_title, headline):
            return True

    # ── Trap 2: Hard-reject non-tech job title ───────────────────────
    for reject_title in _HARD_REJECT_TITLES:
        if reject_title in current_title:
            return True

    # ── Trap 3: Non-tech persona hiding in the summary ───────────────
    _non_tech_personas = [
        "marketing manager", "civil engineer", "accountant",
        "hr manager", "operations manager", "customer support",
        "general manager", "receptionist", "office manager",
        "supply chain", "logistics manager",
    ]
    for persona in _non_tech_personas:
        if persona in summary:
            if not _title_or_headline_is_tech(current_title, headline):
                return True

    # ── Trap 4: Impossible skill proficiency claims ──────────────────
    if _has_impossible_skill_claims(skills):
        return True

    # ── Trap 5: Chronological impossibilities ────────────────────────
    if _has_chronological_impossibilities(career_history):
        return True

    # ── Trap 6: All skills at zero duration ──────────────────────────
    if len(skills) >= 5 and _has_zero_duration_all_skills(skills):
        return True

    return False
