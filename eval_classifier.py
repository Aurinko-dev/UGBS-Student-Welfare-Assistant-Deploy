"""
Evaluation of the triage pipeline.   Run from the project root:

    python eval_classifier.py                       # uses eval_testset.csv
    python eval_classifier.py --testset my_file.csv --out eval_output_after

It runs every labelled message through the SAME steps the live app uses
(see views/settings.py), then compares:

    FULL SYSTEM  = crisis check -> chit-chat check -> neural net (or rules)
                   -> finalize_triage (safety net, urgency floor, escalation)
    BASELINE     = keyword rules only (crisis check, chit-chat check,
                   rule_based_classify), no neural net, no safety net

Test set columns (CSV):
    query            the student's message
    category         the correct category (must match config.CATEGORIES),
                     or CRISIS / CHITCHAT for those two routes
    severity         Low / Medium / High / Critical  (blank for CHITCHAT)
    should_escalate  1 if a person must see it, else 0

Outputs (in ./eval_output/):
    eval_results.csv      one row per message, both systems, right/wrong
    confusion_matrix.csv  full-system category confusion matrix
    eval_report.md        summary tables you can paste into the report
"""
import argparse
from pathlib import Path

import pandas as pd

import config
import neural_classifier
import risk_classifier
import re


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()

CRISIS = "CRISIS"
CHITCHAT = "CHITCHAT"
HIGH_SEVERITIES = {"high", "critical"}
OUT_DIR = Path("eval_output")


# ---------------------------------------------------------------- predictions
def predict_full(query: str) -> dict:
    """Exactly what the live app does for one message."""
    n = risk_classifier.normalize_query(query)
    if risk_classifier.check_crisis(n):
        return dict(category=CRISIS, severity="Critical", escalated=True,
                    reason="crisis", confidence=None, model_category=CRISIS)
    if risk_classifier.is_chitchat(n):
        return dict(category=CHITCHAT, severity="", escalated=False,
                    reason="", confidence=None, model_category=CHITCHAT)

    triage = risk_classifier.rule_based_classify(n)
    if neural_classifier.is_available():
        cat, sev, conf, p_high = neural_classifier.classify_full(n)
    else:
        cat, sev, conf, p_high = triage.category, triage.severity, None, None

    f_cat, f_sev, f_esc, f_reason, _note = risk_classifier.finalize_triage(
        n, cat, sev, conf, p_high)
    return dict(category=f_cat, severity=f_sev, escalated=bool(f_esc),
                reason=f_reason or "", confidence=conf, model_category=cat)


def predict_baseline(query: str) -> dict:
    """Keyword rules only: no neural net, no safety net, no escalation policy.
    A message counts as escalated if the rules call it High/Critical."""
    n = risk_classifier.normalize_query(query)
    if risk_classifier.check_crisis(n):
        return dict(category=CRISIS, severity="Critical", escalated=True)
    if risk_classifier.is_chitchat(n):
        return dict(category=CHITCHAT, severity="", escalated=False)
    t = risk_classifier.rule_based_classify(n)
    return dict(category=t.category, severity=t.severity,
                escalated=str(t.severity).lower() in HIGH_SEVERITIES)


# -------------------------------------------------------------------- metrics
def per_class(y_true, y_pred, labels) -> pd.DataFrame:
    rows = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(dict(category=lab, support=tp + fn, precision=prec, recall=rec, f1=f1))
    return pd.DataFrame(rows)


def escalation_stats(should, did) -> dict:
    tp = sum(1 for s, d in zip(should, did) if s and d)
    fn = sum(1 for s, d in zip(should, did) if s and not d)
    fp = sum(1 for s, d in zip(should, did) if not s and d)
    tn = sum(1 for s, d in zip(should, did) if not s and not d)
    return dict(tp=tp, fn=fn, fp=fp, tn=tn,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                precision=tp / (tp + fp) if tp + fp else 0.0)


def training_overlap(queries) -> list:
    """Test messages that also appear (exact match, ignoring case) in the
    training data. Those inflate accuracy, so they should be replaced."""
    path = Path(config.TRAINING_DATA_PATH)
    if not path.exists():
        return []
    tr = pd.read_csv(path).fillna("")
    col = next((c for c in tr.columns
                if c.lower() in ("query", "text", "question", "message", "utterance")),
               tr.columns[0])
    seen = {str(x).strip().lower() for x in tr[col]}
    return [q for q in queries if str(q).strip().lower() in seen]


def pct(x: float) -> str:
    return f"{x:.1%}"


def md_table(df: pd.DataFrame, pct_cols=()) -> str:
    d = df.copy()
    for c in pct_cols:
        d[c] = d[c].map(pct)
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="eval_testset.csv")
    ap.add_argument("--out", default="eval_output",
                    help="folder for the results (use a new name to keep earlier runs)")
    args = ap.parse_args()
    out_dir = Path(args.out)

    df = pd.read_csv(args.testset).fillna("")
    for col in ("query", "category", "severity", "should_escalate"):
        if col not in df.columns:
            raise SystemExit(f"Test set is missing the '{col}' column.")

    # Match labels to the app's real category names (case-insensitive).
    canon = {c.strip().lower(): c for c in config.CATEGORIES}
    canon[CRISIS.lower()] = CRISIS
    canon[CHITCHAT.lower()] = CHITCHAT
    unknown = sorted({c for c in df["category"] if str(c).strip().lower() not in canon})
    if unknown:
        print("\n!! These labels are not in config.CATEGORIES (they will always count "
              "as wrong; fix the CSV):")
        for u in unknown:
            print("   -", u)
        print("   Valid categories:", ", ".join(config.CATEGORIES), "\n")
    df["category"] = [canon.get(str(c).strip().lower(), c) for c in df["category"]]
    df["should_escalate"] = df["should_escalate"].astype(int)

    leaked = training_overlap(df["query"])
    if leaked:
        print(f"\n!! {len(leaked)} test message(s) also appear in the training data "
              f"({config.TRAINING_DATA_PATH}); replace them or accuracy will look inflated:")
        for q in leaked:
            print("   -", q)
        print()

    print(f"Evaluating {len(df)} messages "
          f"({'neural network' if neural_classifier.is_available() else 'RULE-BASED FALLBACK (network not trained)'})...")

    full = pd.DataFrame([predict_full(q) for q in df["query"]]).add_prefix("full_")
    base = pd.DataFrame([predict_baseline(q) for q in df["query"]]).add_prefix("base_")
    res = pd.concat([df.reset_index(drop=True), full, base], axis=1)

    res["full_cat_ok"] = res["category"] == res["full_category"]
    res["base_cat_ok"] = res["category"] == res["base_category"]
    res["full_sev_ok"] = res["severity"].str.lower() == res["full_severity"].str.lower()
    res["base_sev_ok"] = res["severity"].str.lower() == res["base_severity"].str.lower()
    res["full_esc_ok"] = res["should_escalate"].astype(bool) == res["full_escalated"]
    res["base_esc_ok"] = res["should_escalate"].astype(bool) == res["base_escalated"]

    out_dir.mkdir(exist_ok=True)
    res.to_csv(out_dir / "eval_results.csv", index=False)

    labels = sorted(set(res["category"]) | set(res["full_category"]))
    cm = pd.crosstab(res["category"], res["full_category"]).reindex(
        index=labels, columns=labels, fill_value=0)
    cm.index.name = "actual \\ predicted"
    cm.to_csv(out_dir / "confusion_matrix.csv")

    # Overall comparison
    esc_full = escalation_stats(res["should_escalate"].astype(bool), res["full_escalated"])
    esc_base = escalation_stats(res["should_escalate"].astype(bool), res["base_escalated"])
    overall = pd.DataFrame([
        dict(metric="Category accuracy", baseline=res["base_cat_ok"].mean(),
             full=res["full_cat_ok"].mean()),
        dict(metric="Severity accuracy", baseline=res["base_sev_ok"].mean(),
             full=res["full_sev_ok"].mean()),
        dict(metric="Escalation recall (cases needing a person that were caught)",
             baseline=esc_base["recall"], full=esc_full["recall"]),
        dict(metric="Escalation precision (escalations that were needed)",
             baseline=esc_base["precision"], full=esc_full["precision"]),
    ])

    pc_full = per_class(res["category"], res["full_category"], labels)
    pc_full = pc_full[pc_full["support"] > 0]

    missed = res[(res["should_escalate"] == 1) & (~res["full_escalated"])]
    wrong = res[~res["full_cat_ok"]]

    # ------------------------------------------------------------- console
    print("\n=== OVERALL (baseline = keyword rules only) ===")
    print(overall.assign(baseline=overall["baseline"].map(pct),
                         full=overall["full"].map(pct)).to_string(index=False))
    print(f"\nEscalations missed by full system: {esc_full['fn']} of "
          f"{esc_full['tp'] + esc_full['fn']}")
    for _, r in missed.iterrows():
        print(f"   MISSED: [{r['category']}] {r['query'][:90]}")
    print(f"\nMisclassified by full system: {len(wrong)} of {len(res)}")

    # -------------------------------------------------------------- report
    rep = [
        "# Classifier evaluation",
        f"Test set: `{args.testset}`, {len(res)} messages. "
        f"Model: {'neural network' if neural_classifier.is_available() else 'rule-based fallback'}.",
        "",
        "## Overall: keyword baseline vs full system",
        md_table(overall, pct_cols=("baseline", "full")),
        "",
        "## Escalation detail (full system)",
        f"- Cases that needed a person: {esc_full['tp'] + esc_full['fn']}",
        f"- Caught: {esc_full['tp']}, missed: {esc_full['fn']}",
        f"- Unnecessary escalations: {esc_full['fp']}",
        "",
        "## Per-category results (full system)",
        md_table(pc_full, pct_cols=("precision", "recall", "f1")),
        "",
        "## Confusion matrix (full system; rows = actual, columns = predicted)",
        md_table(cm.reset_index(), ()),
        "",
        "## Missed escalations",
    ]
    rep += ([f"- [{r['category']}] {r['query']}" for _, r in missed.iterrows()]
            or ["None. Every case that needed a person was escalated."])
    rep += ["", "## Training-data overlap (exact matches)",
            f"{len(leaked)} of {len(res)} test messages also appear in the training data."
            + ("" if not leaked else " Replace these: " + "; ".join(f'"{q}"' for q in leaked))]
    rep += ["", "## Misclassified messages (full system)"]
    rep += ([f"- \"{r['query']}\": expected **{r['category']}**, got **{r['full_category']}**"
             for _, r in wrong.iterrows()] or ["None."])
    report_path = out_dir / "eval_report.md"
    report_path.write_text("\n".join(rep), encoding="utf-8")
    print(f"\nSaved: {report_path.resolve()}")
    print(f"Also in {out_dir.resolve()}: eval_results.csv, confusion_matrix.csv")


if __name__ == "__main__":
    main()