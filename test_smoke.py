#!/usr/bin/env python3
"""
Smoke test: generates 150 synthetic candidate records (including honeypots)
and runs the full ZenithRank pipeline to verify correctness.
"""
import json
import gzip
import os
import random
import string
import subprocess
import sys

random.seed(42)

def rand_id():
    return "CAND-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

GOOD_TITLES = [
    "ML Engineer", "Search Engineer", "Data Scientist", "NLP Engineer",
    "Software Engineer - ML", "Applied Scientist", "Ranking Engineer",
    "Recommendation Engineer", "Senior ML Engineer", "AI Engineer",
    "Backend Engineer", "Platform Engineer", "Full Stack Developer",
]

GOOD_COMPANIES = [
    "Google", "Amazon", "Microsoft", "Meta", "Netflix", "Spotify",
    "Uber", "Airbnb", "Stripe", "Flipkart", "Swiggy", "Razorpay",
    "PhonePe", "Ola", "Meesho", "Groww", "Zerodha", "Paytm",
]

ML_SKILLS = [
    "Python", "Machine Learning", "Deep Learning", "NLP",
    "PyTorch", "TensorFlow", "Elasticsearch", "Information Retrieval",
    "Ranking", "Recommendation Systems", "Feature Engineering",
    "Data Pipelines", "SQL", "Docker", "Kubernetes",
]

TRAP_TITLES = [
    "Marketing Manager", "Civil Engineer", "Accountant",
    "HR Manager", "Operations Manager", "Customer Support",
]


def make_good_candidate(idx):
    cid = rand_id()
    yoe = round(random.uniform(4.0, 10.0), 1)
    title = random.choice(GOOD_TITLES)
    company = random.choice(GOOD_COMPANIES)
    skills = random.sample(ML_SKILLS, k=random.randint(4, 8))
    
    return {
        "candidate_id": cid,
        "profile": {
            "summary": f"Experienced {title.lower()} with {yoe} years building production ML systems, "
                       f"search engines, and recommendation infrastructure. Proficient in distributed systems.",
            "headline": f"{title} | ML | Search | NLP",
            "current_title": title,
            "current_company": company,
            "location": random.choice(["Bangalore", "Pune", "Hyderabad", "Mumbai", "Delhi NCR", "Chennai", "Noida"]),
            "years_of_experience": yoe,
        },
        "skills": [
            {
                "name": s,
                "proficiency": random.choice(["expert", "advanced", "intermediate"]),
                "duration_months": random.randint(12, 72),
            }
            for s in skills
        ],
        "career_history": [
            {
                "title": title,
                "company": company,
                "duration_months": random.randint(12, 48),
                "company_age_years": random.randint(5, 30),
                "description": f"Built and maintained ML pipelines, search ranking systems, and recommendation engines.",
            }
        ],
        "education": [
            {
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "institution": random.choice(["IIT Bombay", "IIT Delhi", "BITS Pilani", "NIT Trichy", "VIT"]),
                "tier": random.choice(["tier_1", "tier_2", "tier_3"]),
            }
        ],
        "redrob_signals": {
            "recruiter_response_rate": round(random.uniform(0.2, 0.9), 2),
            "notice_period_days": random.choice([15, 30, 60, 90]),
            "willing_to_relocate": random.choice([True, False]),
        },
        "certifications": [],
        "projects": [],
    }


def make_honeypot_boilerplate(idx):
    """Trap 1: Non-tech title with boilerplate summary."""
    cid = rand_id()
    title = random.choice(TRAP_TITLES)
    return {
        "candidate_id": cid,
        "profile": {
            "summary": f"My professional background is in marketing manager — I've built and led teams "
                       f"across multiple domains. Expert in machine learning, deep learning, NLP, PyTorch.",
            "headline": f"Experienced {title}",
            "current_title": title,
            "current_company": "Generic Corp",
            "location": "Mumbai",
            "years_of_experience": 7.0,
        },
        "skills": [
            {"name": "Machine Learning", "proficiency": "expert", "duration_months": 0},
            {"name": "Deep Learning", "proficiency": "expert", "duration_months": 0},
            {"name": "NLP", "proficiency": "expert", "duration_months": 0},
            {"name": "Python", "proficiency": "expert", "duration_months": 0},
        ],
        "career_history": [
            {"title": title, "company": "Generic Corp", "duration_months": 36, "company_age_years": 10}
        ],
        "education": [{"degree": "MBA", "field_of_study": "Marketing", "institution": "State U", "tier": "tier_3"}],
        "redrob_signals": {"recruiter_response_rate": 0.5, "notice_period_days": 30, "willing_to_relocate": True},
        "certifications": [],
        "projects": [],
    }


def make_honeypot_chronological(idx):
    """Trap 2: Tenure exceeds company age."""
    cid = rand_id()
    return {
        "candidate_id": cid,
        "profile": {
            "summary": "Senior ML engineer with expertise in recommendation systems and search.",
            "headline": "ML Engineer | Search | NLP",
            "current_title": "ML Engineer",
            "current_company": "NewStartup",
            "location": "Bangalore",
            "years_of_experience": 8.0,
        },
        "skills": [
            {"name": "Machine Learning", "proficiency": "expert", "duration_months": 48},
            {"name": "Python", "proficiency": "advanced", "duration_months": 60},
        ],
        "career_history": [
            {
                "title": "ML Engineer",
                "company": "NewStartup",
                "duration_months": 120,  # 10 years
                "company_age_years": 3,   # Company only 3 years old!
            }
        ],
        "education": [{"degree": "B.Tech", "field_of_study": "CS", "institution": "IIT", "tier": "tier_1"}],
        "redrob_signals": {"recruiter_response_rate": 0.7, "notice_period_days": 30, "willing_to_relocate": True},
        "certifications": [],
        "projects": [],
    }


def main():
    output_path = os.path.join(os.path.dirname(__file__), "test_candidates.jsonl.gz")
    
    records = []
    # 120 good candidates
    for i in range(120):
        records.append(make_good_candidate(i))
    # 15 boilerplate honeypots
    for i in range(15):
        records.append(make_honeypot_boilerplate(i))
    # 15 chronological honeypots
    for i in range(15):
        records.append(make_honeypot_chronological(i))
    
    random.shuffle(records)
    
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    
    print(f"Generated {len(records)} test candidates → {output_path}")
    
    # Run the pipeline
    result = subprocess.run(
        [sys.executable, "rank.py", "--candidates", output_path, "--out", "test_submission.csv"],
        capture_output=True, text=True
    )
    print(result.stdout)
    print(result.stderr)
    
    if result.returncode != 0:
        print(f"FAILED with exit code {result.returncode}")
        sys.exit(1)
    
    # Verify output
    import csv
    with open("test_submission.csv", "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    print(f"\nOutput rows: {len(rows)} (header + {len(rows)-1} data)")
    assert len(rows) == 101, f"Expected 101 rows, got {len(rows)}"
    
    # Check no honeypots in output
    honeypot_ids = set()
    for rec in records:
        profile = rec.get("profile", {})
        title = profile.get("current_title", "").lower()
        if title in ["marketing manager", "civil engineer", "accountant", "hr manager", "operations manager", "customer support"]:
            honeypot_ids.add(rec["candidate_id"])
        # chronological trap
        for job in rec.get("career_history", []):
            if job.get("duration_months", 0) / 12.0 > job.get("company_age_years", 999):
                honeypot_ids.add(rec["candidate_id"])
    
    output_ids = {row[0] for row in rows[1:]}
    leaked = output_ids & honeypot_ids
    print(f"Honeypot IDs in pool: {len(honeypot_ids)}")
    print(f"Honeypots leaked into top-100: {len(leaked)}")
    assert len(leaked) == 0, f"HONEYPOTS LEAKED: {leaked}"
    
    print("\n✓ All smoke tests passed!")


if __name__ == "__main__":
    main()
