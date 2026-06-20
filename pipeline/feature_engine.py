#!/usr/bin/env python3
"""
Stage 1: Multiplier Synthesis Engine.

Parses multi-dimensional signals from each candidate profile and collapses
them into a single floating-point *profile multiplier* that modulates the
raw TF-IDF similarity score produced by the retrieval stage.

Dimensions scored:
  • Experience band alignment (5-9 YOE sweet-spot)
  • Corporate DNA (pure consulting/services penalty)
  • Education tier
  • Geographic proximity / relocation willingness
  • Recruiter-intent signals (response rate, notice period)
  • Hard-rejection filters (academic-only, LangChain-wrapper, stale-code,
    pure-CV/Speech/Robotics)
"""

from typing import Any, Dict, List, Set
from pipeline.anti_trap import clean_text_lower


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVICE_GIANTS: Set[str] = frozenset({
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl",
    # Common long-form variants
    "tata consultancy services", "tata consultancy",
    "infosys ltd", "wipro limited", "wipro ltd",
    "accenture solutions", "cognizant technology solutions",
    "capgemini technology services", "hcl technologies",
    "tech mahindra limited",
})

PREFERRED_HUBS: Set[str] = frozenset({
    "pune", "noida", "hyderabad", "mumbai", "delhi ncr",
    "bangalore", "bengaluru", "chennai", "gurgaon", "gurugram",
    "delhi", "new delhi",
})

# Tokens that signal genuine IR / ML / Search / Ranking background
_IR_ML_TOKENS = frozenset([
    "information retrieval", "search engine", "search ranking",
    "dense retrieval", "hybrid retrieval", "vector search",
    "recommendation", "recommender", "ranking system",
    "learning to rank", "ndcg", "mrr", "mean average precision",
    "tf-idf", "bm25", "approximate nearest neighbor", "ann",
    "faiss", "elasticsearch", "solr", "lucene",
    "embedding", "sentence transformer", "bi-encoder", "cross-encoder",
    "machine learning", "deep learning", "nlp",
    "natural language processing", "transformer",
    "pytorch", "tensorflow", "scikit-learn", "xgboost",
    "lightgbm", "catboost", "feature engineering",
    "a/b testing", "online evaluation", "click model",
])

# Pure non-text specialisations (CV, Speech, Robotics)
_NON_TEXT_SPECIALIST_TOKENS = frozenset([
    "computer vision", "image segmentation", "object detection",
    "yolo", "image classification", "pose estimation",
    "speech recognition", "speech synthesis", "tts", "asr",
    "robotics", "robot operating system", "ros", "slam",
    "autonomous driving", "lidar", "point cloud",
])


# ---------------------------------------------------------------------------
# Sub-multiplier functions
# ---------------------------------------------------------------------------

def _experience_modifier(yoe: float) -> float:
    """Sweet-spot: 5-9 YOE → 1.35x; gentle taper outside; hard penalty < 3 or > 13."""
    if 5.0 <= yoe <= 9.0:
        return 1.35
    elif 3.0 <= yoe < 5.0:
        return 1.0
    elif 9.0 < yoe <= 13.0:
        return 1.0
    else:
        return 0.25


def _corporate_dna_modifier(career_history: list) -> float:
    """
    Pure IT-services backgrounds get a severe penalty (0.45x).
    Mixed backgrounds with at least one product company are rewarded (1.25x).
    """
    past_companies: Set[str] = set()
    for job in career_history:
        company = clean_text_lower(job.get("company", ""))
        if company:
            past_companies.add(company)

    if not past_companies:
        return 0.80  # no history is suspicious but not fatal

    is_pure_service = past_companies.issubset(SERVICE_GIANTS)
    return 0.45 if is_pure_service else 1.25


def _education_modifier(education: list) -> float:
    """Tier-1 institutions get a bump; Tier-2 gets a smaller bump."""
    modifier = 1.0
    for edu in education:
        tier = clean_text_lower(edu.get("tier", "unknown"))
        if tier == "tier_1":
            return 1.25  # best possible, return early
        elif tier == "tier_2" and modifier < 1.15:
            modifier = 1.15
    return modifier


def _geo_modifier(location: str, willing_to_relocate: bool) -> float:
    """Candidates in preferred hiring hubs or willing to relocate score 1.0x."""
    for hub in PREFERRED_HUBS:
        if hub in location:
            return 1.0
    return 1.0 if willing_to_relocate else 0.5


def _intent_modifier(response_rate: float, notice_days: int) -> float:
    """
    Low response rate → strong penalty (likely passive/unresponsive).
    Short notice period → bonus (can start sooner).
    """
    modifier = 1.0

    if response_rate < 0.05:
        modifier *= 0.10       # essentially ghosting recruiters
    elif response_rate < 0.15:
        modifier *= 0.20
    elif response_rate >= 0.60:
        modifier *= 1.15       # actively engaging

    if notice_days <= 15:
        modifier *= 1.35
    elif notice_days <= 30:
        modifier *= 1.25
    elif notice_days > 90:
        modifier *= 0.60
    elif notice_days > 60:
        modifier *= 0.75

    return modifier


def _hard_reject_modifier(candidate: Dict[str, Any]) -> float:
    """
    Returns 0.0 (instant elimination) for candidates matching any of the
    five hard-rejection rules from Section 1:

    R2: Pure academic researchers with zero production deployment history.
    R3: LangChain tutorial wrappers (< 12 months AI, no pre-LLM ML).
    R4: Senior engineers who haven't shipped code in the last 18 months.
    R5: Pure CV / Speech / Robotics specialists with no text retrieval.
    (R1 — pure services — is handled by _corporate_dna_modifier with a
     severe penalty rather than a hard zero, to avoid double-filtering.)
    """
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    summary = clean_text_lower(profile.get("summary", ""))
    headline = clean_text_lower(profile.get("headline", ""))
    current_title = clean_text_lower(profile.get("current_title", ""))
    combined_text = f"{summary} {headline} {current_title}"

    # ── R2: Pure academic, zero production ───────────────────────────
    is_academic = any(
        kw in combined_text
        for kw in ["professor", "postdoc", "phd candidate", "research fellow",
                    "research associate", "doctoral"]
    )
    has_production = any(
        kw in combined_text
        for kw in ["production", "deployed", "shipped", "pipeline",
                    "microservice", "api", "backend", "infrastructure",
                    "scale", "latency", "sre", "devops"]
    )
    if is_academic and not has_production:
        # Also check career history for any non-academic employer
        non_academic_employers = [
            job for job in career_history
            if not any(kw in clean_text_lower(job.get("company", ""))
                       for kw in ["university", "institute", "research lab",
                                  "college", "academia"])
        ]
        if len(non_academic_employers) == 0:
            return 0.0

    # ── R3: LangChain wrapper with no ML depth ───────────────────────
    ai_months = 0
    has_pre_llm_ml = False
    for skill in skills:
        skill_name = clean_text_lower(skill.get("name", ""))
        duration = 0
        try:
            duration = int(skill.get("duration_months", 0))
        except (ValueError, TypeError):
            pass
        if any(kw in skill_name for kw in ["ai", "machine learning", "ml",
                                            "deep learning", "nlp", "llm",
                                            "langchain", "gpt"]):
            ai_months += duration
        if any(kw in skill_name for kw in ["scikit", "sklearn", "xgboost",
                                            "tensorflow", "pytorch", "keras",
                                            "feature engineering", "random forest",
                                            "gradient boosting", "svm"]):
            has_pre_llm_ml = True

    if ai_months < 12 and not has_pre_llm_ml:
        if "langchain" in combined_text or "llm wrapper" in combined_text:
            return 0.0

    # ── R4: Senior/Architect without recent code ─────────────────────
    is_senior_or_architect = any(
        kw in current_title
        for kw in ["senior", "staff", "principal", "architect", "director",
                    "vp", "head of"]
    )
    if is_senior_or_architect:
        # Check if last coding role is within 18 months
        signals = candidate.get("redrob_signals", {})
        months_since_last_code = None
        try:
            months_since_last_code = int(signals.get("months_since_last_code_commit", -1))
        except (ValueError, TypeError):
            pass
        if months_since_last_code is not None and months_since_last_code > 18:
            return 0.0

    # ── R5: Pure CV / Speech / Robotics specialist ───────────────────
    non_text_score = sum(
        1 for token in _NON_TEXT_SPECIALIST_TOKENS if token in combined_text
    )
    text_retrieval_score = sum(
        1 for token in _IR_ML_TOKENS if token in combined_text
    )
    # Also count from skill names
    for skill in skills:
        skill_name = clean_text_lower(skill.get("name", ""))
        if any(t in skill_name for t in ["computer vision", "image", "speech",
                                          "robotics", "ros", "slam", "lidar"]):
            non_text_score += 1
        if any(t in skill_name for t in ["search", "retrieval", "nlp",
                                          "ranking", "recommendation",
                                          "text", "embedding"]):
            text_retrieval_score += 1

    if non_text_score >= 3 and text_retrieval_score == 0:
        return 0.0

    return 1.0  # pass all hard-rejection gates


# ---------------------------------------------------------------------------
# Skill-depth bonus
# ---------------------------------------------------------------------------

def _skill_depth_bonus(skills: list) -> float:
    """
    Reward candidates who have deep, sustained experience in IR/ML-adjacent
    skills (measured by cumulative months in relevant skill categories).
    """
    relevant_months = 0
    for skill in skills:
        skill_name = clean_text_lower(skill.get("name", ""))
        duration = 0
        try:
            duration = int(skill.get("duration_months", 0))
        except (ValueError, TypeError):
            pass
        if any(kw in skill_name for kw in [
            "python", "machine learning", "ml", "deep learning",
            "nlp", "search", "retrieval", "ranking", "recommendation",
            "pytorch", "tensorflow", "elasticsearch", "solr",
            "embedding", "transformer", "data science", "ai",
            "information retrieval", "vector", "faiss",
        ]):
            relevant_months += duration

    if relevant_months >= 60:
        return 1.30
    elif relevant_months >= 36:
        return 1.15
    elif relevant_months >= 18:
        return 1.05
    return 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_profile_multipliers(candidate: Dict[str, Any]) -> float:
    """
    Collapse all candidate dimensions into a single floating-point multiplier.
    Returns 0.0 for hard-rejected candidates.

    The multiplier is the product of:
      exp × dna × edu × geo × intent × hard_reject × skill_depth
    """
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    # ── Hard rejection (returns 0.0 → eliminates candidate) ──────────
    hard_reject = _hard_reject_modifier(candidate)
    if hard_reject == 0.0:
        return 0.0

    # ── Continuous multipliers ───────────────────────────────────────
    yoe = 0.0
    try:
        yoe = float(profile.get("years_of_experience", 0))
    except (ValueError, TypeError):
        pass

    location = clean_text_lower(profile.get("location", ""))
    willing = bool(signals.get("willing_to_relocate", False))

    response_rate = 0.0
    try:
        response_rate = float(signals.get("recruiter_response_rate", 0.0))
    except (ValueError, TypeError):
        pass

    notice_days = 90
    try:
        notice_days = int(signals.get("notice_period_days", 90))
    except (ValueError, TypeError):
        pass

    exp = _experience_modifier(yoe)
    dna = _corporate_dna_modifier(career_history)
    edu = _education_modifier(education)
    geo = _geo_modifier(location, willing)
    intent = _intent_modifier(response_rate, notice_days)
    skill_depth = _skill_depth_bonus(skills)

    return exp * dna * edu * geo * intent * skill_depth
