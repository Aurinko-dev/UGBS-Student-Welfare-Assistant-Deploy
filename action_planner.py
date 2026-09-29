"""
This is what separates an agent from a chatbot: after classifying the
issue, the system doesn't just answer -- it decides WHERE the student
should actually go and generates a concrete next-steps plan.

The office mapping is deterministic (these are policy facts -- the system
should never let an LLM guess which office handles what), while the
step-by-step plan is generated per-query since it depends on what the
student actually asked (llm_engine.generate_action_plan).
"""

OFFICE_MAP = {
    "Financial Distress": {
        "office": "Student Financial Aid Office (SFAO)",
        "note": "Handles fee waivers, scholarships, and financial hardship applications.",
    },
    "Academic Distress": {
        "office": "Academic Affairs Directorate / your Departmental Academic Advisor",
        "note": "Handles deferment, withdrawal, probation, and course-load issues.",
    },
    "Accommodation": {
        "office": "Traditional Halls / UGEL Hostels Office / Private hostel management",
        "note": "Handles room allocation, roommate issues, and hall/hostel facility matters.",
    },
    "Sexual Harassment / GBV": {
        "office": "CEGENSA (Centre for Gender Studies and Advocacy)",
        "note": "Handles sexual harassment and misconduct reports, both informal and "
                "formal, and runs a dedicated crisis and counselling unit. "
                "https://cegensa.ug.edu.gh -- the Careers and Counselling Directorate "
                "is also available for confidential support alongside or instead of "
                "a formal complaint.",
    },
    "Mental Health / Counselling": {
        "office": "University of Ghana Counselling and Placement Centre (UGCCD)",
        "note": "Handles counselling appointments and mental health support.",
    },
    "Career Guidance": {
        "office": "Careers and Counselling Directorate",
        "note": "Handles CV reviews, mock interviews, and internship/job placement support.",
    },
    "Student Governance / JCR": {
        "office": "Class Representative -> BHJCR Council of Representatives and Welfare Committee",
        "note": "For UGBS-specific welfare issues, class reps escalate to the BHJCR "
                "Welfare Committee and General Assembly. Disputes about BHJCR officer "
                "conduct or funds go to the BHJCR Judiciary; election disputes go to the "
                "BHJCR Electoral Commission (or the university-wide SRC Electoral "
                "Commission for SRC-level elections). For formal disciplinary matters at "
                "the university level (not just BHJCR), the process runs through the "
                "Disciplinary Board for Junior Members, per the University Statutes.",
    },
}


_DEFAULT = {
    "office": "UGBS Academic Office",
    "note": "General point of contact when the issue doesn't fit a specific "
            "category above, or the system could not confidently classify it. "
            "The Academic Office can redirect you to the right place.",
}


def get_recommended_office(category: str):
    """Always returns a dict with 'office' and 'note' -- never None -- so
    callers don't need a separate None-check before showing a referral."""
    return OFFICE_MAP.get(category, _DEFAULT)
