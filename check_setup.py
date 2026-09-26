"""
Setup checker.   Run from the project root (the folder with app.py):

    python check_setup.py

It checks that the files are in the right places, that you are running the
NEW versions of the files we changed, that every .py file has valid syntax,
that the modules import, that the test sets are valid, and that the crisis
rules behave. Nothing is modified. Every line is [OK], [!!] (warning) or
[XX] (problem to fix).
"""
import ast
import csv
import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path.cwd()
results = {"ok": 0, "warn": 0, "fail": 0}


def report(level: str, msg: str) -> None:
    tag = {"ok": "[OK]", "warn": "[!!]", "fail": "[XX]"}[level]
    results[level] += 1
    print(f"{tag} {msg}")


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8", errors="replace")


# ------------------------------------------------------------------ 1. location
section("1. Location")
if (ROOT / "app.py").exists() and (ROOT / "config.py").exists():
    report("ok", f"Running from the project root: {ROOT}")
else:
    report("fail", f"app.py / config.py not found in {ROOT}. "
                   "cd into the project folder first, then run this again.")
    sys.exit(1)

# ---------------------------------------------------------------- 2. files exist
section("2. Required files")
REQUIRED = [
    "app.py", "config.py", "logo.py", "ui_theme.py", "analytics_db.py",
    "risk_classifier.py", "neural_classifier.py", "classifier_model.py",
    "train_classifier.py", "eval_classifier.py",
    "eval_testset.csv", "eval_testset_crisis_v2.csv", "eval_testset_crisis_v3.csv",
    "views/user_interface.py", "views/admin_interface.py",
    "views/live_admin_interface.py", "views/settings.py",
    "assets/ugbs_logo.png",
    "program_guidance.py",
    "ugbs_programmes_and_options.md", "ugbs_offices_and_contacts.md",
]
for f in REQUIRED:
    if (ROOT / f).exists():
        report("ok", f)
    else:
        report("fail", f"MISSING: {f}")

# ------------------------------------------------- 3. are these the NEW versions?
section("3. New versions of the files we changed")
MARKERS = [
    ("eval_classifier.py", "--out", "has the --out option"),
    ("risk_classifier.py", '"killing myself"', "has the broader crisis phrases"),
    ("risk_classifier.py", "possible_crisis", "has the second-tier crisis escalation"),
    ("views/live_admin_interface.py", "possible_crisis", "shows the possible-crisis reason"),
    ("views/user_interface.py", "program_guidance", "has the programme guidance"),
    ("config.py", "ugbs_programmes_and_options.md", "lists the new knowledge-base documents"),
    ("app.py", "from logo import logo_html", "uses the shared logo helper"),
    ("views/user_interface.py", "logo_html(", "shows the logo in the header"),
    ("views/live_admin_interface.py", 'stForm', "has the blue Acknowledge styling"),
]
for path, needle, meaning in MARKERS:
    if not (ROOT / path).exists():
        continue
    if needle in read(path):
        report("ok", f"{path} {meaning}")
    else:
        report("fail", f"{path} looks like an OLD copy (does not {meaning}). "
                       "Replace it with the latest version.")

if (ROOT / "user_interface.py").exists():
    report("warn", "A user_interface.py sits in the project root. The app reads "
                   "views/user_interface.py, so a copy here does nothing.")

# ------------------------------------------------------------------ 4. syntax
section("4. Python syntax")
bad = 0
for p in sorted(ROOT.rglob("*.py")):
    if any(part in {"venv", ".venv", "__pycache__", "site-packages"} for part in p.parts):
        continue
    try:
        ast.parse(p.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as e:
        bad += 1
        report("fail", f"{p.relative_to(ROOT)} line {e.lineno}: {e.msg}")
if not bad:
    report("ok", "Every .py file parses without syntax errors")

# ----------------------------------------------------------------- 5. packages
section("5. Packages")
for pkg in ("streamlit", "pandas", "plotly", "torch", "sklearn", "dotenv",
            "langchain_huggingface", "langchain_community"):
    if importlib.util.find_spec(pkg):
        report("ok", pkg)
    else:
        report("warn" if pkg in {"sklearn"} else "fail", f"not installed: {pkg}")
try:
    import streamlit as st
    ver = tuple(int(x) for x in st.__version__.split(".")[:2])
    if ver < (1, 36):
        report("fail", f"Streamlit {st.__version__} is too old for st.Page / st.navigation "
                       "(needs 1.36+). Run: pip install --upgrade streamlit")
    else:
        report("ok", f"Streamlit version {st.__version__}")
except Exception as e:  # noqa: BLE001
    report("fail", f"Could not read the Streamlit version: {e}")

# ------------------------------------------------------------------- 6. imports
section("6. Modules load")
sys.path.insert(0, str(ROOT))
config = risk = neural = None
try:
    config = importlib.import_module("config")
    report("ok", f"config ({len(config.CATEGORIES)} categories)")
except Exception as e:  # noqa: BLE001
    report("fail", f"config failed to import: {e}")
try:
    risk = importlib.import_module("risk_classifier")
    report("ok", "risk_classifier")
except Exception as e:  # noqa: BLE001
    report("fail", f"risk_classifier failed to import: {e}")
try:
    importlib.import_module("analytics_db")
    report("ok", "analytics_db")
except Exception as e:  # noqa: BLE001
    report("fail", f"analytics_db failed to import: {e}")
try:
    neural = importlib.import_module("neural_classifier")
    if neural.is_available():
        report("ok", "neural_classifier (trained model found)")
    else:
        report("warn", "neural_classifier loads but the model is NOT trained "
                       "(run train_classifier.py). The app uses keyword rules only.")
except Exception as e:  # noqa: BLE001
    report("fail", f"neural_classifier failed to import: {e}")
try:
    logo = importlib.import_module("logo")
    html = logo.logo_html(52)
    if "base64" in html:
        report("ok", "logo.py loads assets/ugbs_logo.png")
    else:
        report("warn", "logo.py is using the text fallback (logo image not readable)")
except Exception as e:  # noqa: BLE001
    report("fail", f"logo failed to import: {e}")

# --------------------------------------------------------------- 7. data + model
section("7. Data, model and knowledge base")
if config:
    for label, path in (("Training data", config.TRAINING_DATA_PATH),
                        ("Classifier weights", config.CLASSIFIER_MODEL_PATH),
                        ("Classifier labels", config.CLASSIFIER_LABELS_PATH)):
        if (ROOT / path).exists():
            report("ok", f"{label}: {path}")
        else:
            report("warn", f"{label} not found: {path}")
    missing = [f for f in config.MARKDOWN_FILES if not (ROOT / f).exists()]
    if missing:
        for f in missing:
            report("fail", f"Knowledge-base file not found: {f}")
    else:
        report("ok", f"All {len(config.MARKDOWN_FILES)} knowledge-base files present")

# ---------------------------------------------------------------- 8. test sets
section("8. Evaluation test sets")
valid = set()
if config:
    valid = {c.strip().lower() for c in config.CATEGORIES} | {"crisis", "chitchat"}
train_set = set()
if config and (ROOT / config.TRAINING_DATA_PATH).exists():
    try:
        import pandas as pd
        tr = pd.read_csv(ROOT / config.TRAINING_DATA_PATH).fillna("")
        col = next((c for c in tr.columns if c.lower() in
                    ("query", "text", "question", "message", "utterance")), tr.columns[0])
        train_set = {str(x).strip().lower() for x in tr[col]}
    except Exception as e:  # noqa: BLE001
        report("warn", f"Could not read the training data: {e}")

for name in ("eval_testset.csv", "eval_testset_crisis_v2.csv", "eval_testset_crisis_v3.csv"):
    path = ROOT / name
    if not path.exists():
        continue
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:  # noqa: BLE001
        report("fail", f"{name} could not be read: {e}")
        continue
    need = {"query", "category", "severity", "should_escalate"}
    if not rows or not need.issubset(rows[0].keys()):
        report("fail", f"{name} is missing columns (needs {sorted(need)})")
        continue
    report("ok", f"{name}: {len(rows)} messages")
    if valid:
        unknown = sorted({r["category"] for r in rows
                          if r["category"].strip().lower() not in valid})
        if unknown:
            report("fail", f"{name} has labels not in config.CATEGORIES: {unknown}")
    dup = len(rows) - len({r["query"].strip().lower() for r in rows})
    if dup:
        report("warn", f"{name} has {dup} duplicate message(s)")
    leak = [r["query"] for r in rows if r["query"].strip().lower() in train_set]
    if leak:
        report("warn", f"{name}: {len(leak)} message(s) also in the training data "
                       f"(inflates accuracy): {leak[:3]}")
    else:
        report("ok", f"{name}: no overlap with the training data")

# -------------------------------------------------------------- 9. crisis rules
section("9. Crisis rules")
if risk:
    must_catch = ["I want to end my life", "I am thinking of killing myself",
                  "I feel like harming myself", "Nobody would miss me if I was gone"]
    must_not = ["How do I cut my expenses this semester",
                "My lecturer is killing me with assignments, how do I ask for an extension",
                "How do I book a counselling session"]
    missed = [q for q in must_catch if not risk.check_crisis(q)]
    false_alarm = [q for q in must_not if risk.check_crisis(q)]
    if missed:
        report("fail", f"Crisis check MISSED: {missed}")
    else:
        report("ok", "Crisis check catches the known crisis messages")
    if false_alarm:
        report("fail", f"Crisis check wrongly fired on: {false_alarm}")
    else:
        report("ok", "Crisis check ignores everyday phrases")

# ---------------------------------------------------------------------- summary
print("\n" + "=" * 52)
print(f"  OK: {results['ok']}    Warnings: {results['warn']}    Problems: {results['fail']}")
if results["fail"]:
    print("  Fix every [XX] line first, then run this again.")
elif results["warn"]:
    print("  No problems. Look at the [!!] warnings and decide if they matter.")
else:
    print("  Everything looks good.")
print("=" * 52)
sys.exit(1 if results["fail"] else 0)