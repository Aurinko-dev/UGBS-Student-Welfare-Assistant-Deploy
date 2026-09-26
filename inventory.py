"""
Project inventory + diagnostic report.

Run from the project root:   python inventory.py > inventory_report.txt

Then upload inventory_report.txt. This is deliberately more than a file
listing -- it specifically checks for the two bug classes that have bitten
this project before:
  1. A .md knowledge-base file that exists on disk but isn't in
     config.MARKDOWN_FILES (silently never indexed).
  2. The vector store (ugbs_welfare_db/) being OLDER than the newest
     knowledge-base file (edited content that was never re-indexed).
"""
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

# Windows' default console encoding (cp1252) can't handle emoji in filenames
# like "1_📊_Admin_Interface.py" -- force UTF-8 output so redirecting to a
# file (or printing to a UTF-8-aware terminal) doesn't crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(".").resolve()
W = 90


def hr(title=""):
    print("\n" + "=" * W)
    if title:
        print(title)
        print("=" * W)


def mtime(p: Path) -> datetime:
    return datetime.fromtimestamp(p.stat().st_mtime)


# --------------------------------------------------------------------- files
hr("1. FILE INVENTORY (.py and .md, excluding venv)")
rows = []
for p in ROOT.rglob("*"):
    if p.is_file() and p.suffix in (".py", ".md") and "venv" not in p.parts:
        rows.append(p)
rows.sort(key=lambda p: str(p))
for p in rows:
    rel = p.relative_to(ROOT)
    size = p.stat().st_size
    print(f"  {str(rel):55s} {size:>8,d} B   {mtime(p):%Y-%m-%d %H:%M}")

# ------------------------------------------------------------------- config
hr("2. CONFIG.PY — categories and knowledge-base file list")
try:
    sys.path.insert(0, str(ROOT))
    import config
    print("Categories (%d):" % len(config.CATEGORIES))
    for c in config.CATEGORIES:
        print(f"   - {c}")
    print(f"\nMARKDOWN_FILES ({len(config.MARKDOWN_FILES)} entries):")
    missing = []
    for f in config.MARKDOWN_FILES:
        exists = os.path.exists(f)
        print(f"   [{'OK' if exists else 'MISSING'}]  {f}")
        if not exists:
            missing.append(f)
    if missing:
        print(f"\n!! {len(missing)} file(s) listed in config.py but not found on disk.")
except Exception as e:
    print(f"Could not import config.py: {e}")
    config = None

# ------------------------------------------- knowledge-base files not registered
hr("3. .md FILES ON DISK NOT IN config.MARKDOWN_FILES (never indexed)")
_NOT_KB = {"README.md", "report_support_material.md", "STAGE4_CHANGELOG.md",
           "STAGE5_CHANGELOG.md", "STAGE5_VALIDATION_RESULTS.md"}  # documentation, not KB content
if config is not None:
    root_md = {p.name for p in ROOT.glob("*.md")}
    registered = set(config.MARKDOWN_FILES)
    unregistered = sorted(root_md - registered - _NOT_KB)
    if unregistered:
        for f in unregistered:
            print(f"   !! {f}  <-- exists but NOT in config.MARKDOWN_FILES")
        print("\n   Fix: add these to MARKDOWN_FILES in config.py, then rerun "
              "build_vectorstore.py.")
    else:
        print("   None -- every root-level .md file (excluding README.md, "
              "report_support_material.md, and the STAGE4/5 changelog/validation docs, "
              "none of which are knowledge-base content) is registered.")

# --------------------------------------------------------- vector store staleness
hr("4. VECTOR STORE FRESHNESS")
db_dir = ROOT / "ugbs_welfare_db"
if config is not None and db_dir.exists():
    kb_files = [ROOT / f for f in config.MARKDOWN_FILES if (ROOT / f).exists()]
    if kb_files:
        newest_kb = max(kb_files, key=lambda p: p.stat().st_mtime)
        db_sqlite = db_dir / "chroma.sqlite3"
        if db_sqlite.exists():
            db_time = mtime(db_sqlite)
            kb_time = mtime(newest_kb)
            print(f"   Vector store last built:     {db_time:%Y-%m-%d %H:%M}")
            print(f"   Newest knowledge-base file:  {kb_time:%Y-%m-%d %H:%M}  ({newest_kb.name})")
            if kb_time > db_time:
                print("\n   !! STALE: a knowledge-base file was edited AFTER the vector "
                      "store was last built.")
                print("      Fix: run `python build_vectorstore.py` again.")
            else:
                print("\n   OK: vector store is newer than all knowledge-base files.")
        else:
            print("   ugbs_welfare_db/ exists but has no chroma.sqlite3 -- looks broken, rebuild it.")
else:
    print("   ugbs_welfare_db/ not found -- run `python build_vectorstore.py`.")

# --------------------------------------------------------------- training data
hr("5. TRAINING DATA")
csv_path = ROOT / "data" / "welfare_training_examples.csv"
if csv_path.exists():
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"   {len(rows)} labeled examples")
    cats = {}
    for r in rows:
        cats[r.get("category", "?")] = cats.get(r.get("category", "?"), 0) + 1
    for c, n in sorted(cats.items()):
        print(f"      {n:4d}  {c}")
else:
    print("   data/welfare_training_examples.csv not found.")

# --------------------------------------------------------------- eval outputs
hr("6. EVALUATION RESULTS ON DISK")
eval_dirs = sorted(p for p in ROOT.glob("eval_output*") if p.is_dir())
if eval_dirs:
    for d in eval_dirs:
        report = d / "eval_report.md"
        if report.exists():
            print(f"\n   --- {d.name} ({mtime(report):%Y-%m-%d %H:%M}) ---")
            lines = report.read_text(encoding="utf-8").splitlines()
            # print just the overall metrics table
            capture = False
            for line in lines:
                if line.startswith("## Overall"):
                    capture = True
                    continue
                if capture:
                    if line.startswith("## "):
                        break
                    if line.strip():
                        print("   " + line)
else:
    print("   No eval_output* folders found.")

# ------------------------------------------------------ risk_classifier keywords
hr("7. RISK_CLASSIFIER.PY — keyword lists (first 200 chars of each)")
rc_path = ROOT / "risk_classifier.py"
if rc_path.exists():
    text = rc_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("_") and "KEYWORDS" in stripped and "=" in stripped:
            print("   " + stripped[:200])
else:
    print("   risk_classifier.py not found.")

hr("DONE")
print("Upload the full output of this script (redirect to a file: "
      "python inventory.py > inventory_report.txt) and send inventory_report.txt.")