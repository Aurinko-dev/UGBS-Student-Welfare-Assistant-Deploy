"""
Central configuration for the UGBS Student Welfare Assistant.

All secrets are read from environment variables (or a local .env file via
python-dotenv). Nothing is hard-coded, so this is safe to commit.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # loads a local .env file if present, no-op otherwise

DB_DIR = "./ugbs_welfare_db"
ANALYTICS_DB_PATH = "./welfare_analytics.sqlite3"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TRAINING_DATA_PATH = "./data/welfare_training_examples.csv"
CLASSIFIER_MODEL_PATH = "./models/welfare_classifier.pt"
CLASSIFIER_LABELS_PATH = "./models/welfare_classifier_labels.json"

# Which LLM generates the final grounded answer. "ollama" runs fully local
# and free (what was used in class) and needs no API key at all; "gemini"
# and "anthropic" are cloud alternatives if you'd rather not run a local model.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

MARKDOWN_FILES = [
    "careers_and_counselling_faq.md",
    "sfao_financial_aid.md",
    "ugbs_academic_policies.md",
    "ugccd counselling policy.md",
    "bhjcr_constitution.md",
    "ug_statutes_governance.md",
    "ug_src_electoral_ci24.md",
    "ug_academic_affairs_qna.md",
    "ug_discipline_governance.md",
    "accommodation.md",
    "sexual_harassment_support.md",
    "ugbs_programmes_and_options.md",
    # Locations, department contacts, and campus navigation/landmarks are
    # all ONE file (not split across several), after several rounds of
    # duplicated/drifting location research kept producing separate files
    # that overlapped and disagreed with each other. See this file's own
    # closing section for exactly what is confirmed vs. still unresolved.
    "ugbs_offices_and_contacts.md",
    # STS (sts.ug.edu.gh) FAQs, split by topic so each file retrieves cleanly.
    "sts_portal_and_payments_faqs.md",
    "sts_academic_records_and_regulations_faqs.md",
    "sts_graduation_and_certificates_faqs.md",
]

CATEGORIES = [
    "Financial Distress",
    "Academic Distress",
    "Accommodation",
    "Mental Health / Counselling",
    "Career Guidance",
    "Student Governance / JCR",
    "Sexual Harassment / GBV",
    "Out of Scope",
]

# Categories where the answer genuinely differs by nationality (fees, some
# scholarships) -- the assistant should ask rather than assume or guess.
NATIONALITY_SENSITIVE_CATEGORIES = ["Financial Distress", "Accommodation"]

# Fallback links for when a question genuinely isn't covered by the indexed
# knowledge base -- used instead of a dead-end "I don't know" response.
UGBS_WEBSITE = "https://ugbs.ug.edu.gh"
UG_WEBSITE = "https://www.ug.edu.gh"

SEVERITY_LEVELS = ["Low", "Medium", "High", "Critical"]


def llm_is_configured() -> bool:
    """Whether the configured provider is actually usable right now.
    Ollama needs no key, just a running local server, so we optimistically
    say yes and let llm_engine's error handling catch it if the server
    isn't up (falls back to template mode automatically)."""
    if LLM_PROVIDER == "ollama":
        return True
    if LLM_PROVIDER == "gemini":
        return bool(GEMINI_API_KEY)
    if LLM_PROVIDER == "anthropic":
        return bool(ANTHROPIC_API_KEY)
    return False