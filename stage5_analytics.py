"""Stage 5 administrative analytics for the UGBS welfare triage prototype.

Run from the project root:
    python stage5_analytics.py
    python stage5_analytics.py --out stage5_analytics_output

This reads the SQLite interaction table and produces CSV summaries suitable
for the Stage 5 report/dashboard. It does not expose or print raw student
messages unless explicitly requested with --include-sensitive.
"""
import argparse
from pathlib import Path
import pandas as pd
import analytics_db


def build_summaries(df: pd.DataFrame):
    if df.empty:
        return {}
    out = {}
    out["category_demand"] = analytics_db.category_counts(df)
    out["severity_demand"] = (
        df["severity"].value_counts().rename_axis("severity").reset_index(name="count")
    )
    out["daily_trend"] = analytics_db.daily_trend(df)
    out["escalation_summary"] = pd.DataFrame([{
        "total_interactions": len(df),
        "escalated_cases": int(df["escalated"].sum()),
        "escalation_rate_pct": analytics_db.escalation_rate(df),
    }])
    if "recommended_office" in df:
        out["office_referrals"] = (
            df[df["recommended_office"].fillna("") != ""]
            ["recommended_office"].value_counts()
            .rename_axis("office").reset_index(name="count")
        )
    if "acknowledged" in df:
        out["workflow_status"] = pd.DataFrame([{
            "escalated": int(df["escalated"].sum()),
            "acknowledged": int(((df["escalated"] == 1) & (df["acknowledged"].fillna(0) == 1)).sum()),
            "unacknowledged_escalations": int(len(analytics_db.get_escalation_queue(df))),
            "unresolved_review_items": int(len(analytics_db.get_unresolved(df))),
        }])
        avg = analytics_db.avg_minutes_to_acknowledge(df)
        out["response_time"] = pd.DataFrame([{
            "average_minutes_to_acknowledge": avg
        }])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stage5_analytics_output")
    ap.add_argument("--include-sensitive", action="store_true",
                    help="include the raw interaction export; use only in a controlled environment")
    args = ap.parse_args()

    analytics_db.init_db()
    df = analytics_db.load_all()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print("No interactions are currently stored in the SQLite database.")
        return

    for name, summary in build_summaries(df).items():
        summary.to_csv(outdir / f"{name}.csv", index=False)

    if args.include_sensitive:
        df.to_csv(outdir / "raw_interactions_sensitive.csv", index=False)
    else:
        safe_cols = [c for c in df.columns if c != "query"]
        df[safe_cols].to_csv(outdir / "interaction_metadata.csv", index=False)

    print(f"Analytics generated for {len(df)} recorded interactions in {outdir.resolve()}")
    print(f"Escalation rate: {analytics_db.escalation_rate(df):.1f}%")


if __name__ == "__main__":
    main()
