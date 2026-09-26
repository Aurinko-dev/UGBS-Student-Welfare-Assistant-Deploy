"""Stage 4 smoke tests for the implemented welfare triage agent.

These tests exercise the deterministic parts of the agent pipeline without
requiring an LLM server, vector database, or trained neural-network weights.
Run: python stage4_agent_tests.py
"""
from action_planner import get_recommended_office
from risk_classifier import check_crisis, finalize_triage, normalize_query, rule_based_classify

CASES = [
    ("I cannot afford my fees this semester.", "Financial Distress"),
    ("I am struggling with my courses and may fail.", "Academic Distress"),
    ("I have an accommodation problem and nowhere to stay.", "Accommodation"),
    ("I feel overwhelmed and need someone to talk to.", "Mental Health / Counselling"),
    ("I need help with my career and internship.", "Career Guidance"),
    ("I need help with my JCR welfare issue.", "Student Governance / JCR"),
    ("I was sexually harassed and do not know who to tell.", "Sexual Harassment / GBV"),
]

for text, expected in CASES:
    normalized = normalize_query(text)
    base = rule_based_classify(normalized)
    category, severity, escalated, reason, note = finalize_triage(
        normalized, base.category, base.severity, None, None
    )
    assert category == expected, f"Expected {expected!r}, got {category!r} for {text!r}"
    assert get_recommended_office(category)["office"], f"No office for {category}"
    print(f"PASS | {category} | {severity} | escalated={escalated} | {text}")

crisis = "I am thinking of killing myself and need help."
assert check_crisis(crisis), "Crisis message was not detected"
print("PASS | Crisis detection")

print("\nStage 4 deterministic agent smoke tests passed.")
