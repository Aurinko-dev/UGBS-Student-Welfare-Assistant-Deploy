"""
Every interaction the chatbot handles gets logged here. This is the whole
of the assignment's "analytical decision-support" requirement (10%): the
admin dashboard reads from this table to show category demand, trends, and
escalation rates — the thing a welfare-office administrator would actually
use to decide where to put resources.
"""
import sqlite3
from datetime import datetime
from typing import Optional
import pandas as pd
import config


def _connect():
    return sqlite3.connect(config.ANALYTICS_DB_PATH)


def init_db():
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            category TEXT NOT NULL,
            severity TEXT NOT NULL,
            escalated INTEGER NOT NULL,
            top_source TEXT,
            recommended_office TEXT,
            classifier_source TEXT
        )
    """)
    # Migration guard for databases created before these columns existed.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(interactions)")}
    for col, coltype in (("recommended_office", "TEXT"), ("classifier_source", "TEXT"),
                         ("needs_review", "INTEGER DEFAULT 0"), ("resolved", "INTEGER DEFAULT 0"),
                         ("admin_note", "TEXT"), ("acknowledged", "INTEGER DEFAULT 0"),
                         ("acknowledged_at", "TEXT"), ("ack_note", "TEXT"),
                         # Previously only appended as free text inside
                         # classifier_source (e.g. "... | escalated: possible_crisis"),
                         # which made it impossible to filter/sort the queue by why a
                         # case was escalated without parsing a log string. Its own
                         # column lets the admin dashboard group/filter cases by reason
                         # (e.g. surface "crisis" and "possible_crisis" ahead of
                         # "low_confidence_sensitive").
                         ("escalation_reason", "TEXT"),
                         # Student-given thumbs up/down on a fully-answered question.
                         # NULL/empty means no feedback was given -- distinct from a
                         # "down" vote, so the admin dashboard doesn't need to guess
                         # whether silence means "fine" or "never asked".
                         ("feedback", "TEXT")):
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE interactions ADD COLUMN {col} {coltype}")
    conn.commit()
    conn.close()


def log_interaction(query: str, category: str, severity: str, escalated: bool,
                     top_source: str = "", recommended_office: str = "",
                     classifier_source: str = "", needs_review: bool = False,
                     escalation_reason: str = "") -> int:
    """Returns the new row's id, so the caller (views/user_interface.py) can
    attach feedback buttons to this specific interaction later in the same
    chat session -- see log_feedback() below."""
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO interactions (timestamp, query, category, severity, escalated, "
        "top_source, recommended_office, classifier_source, needs_review, resolved, "
        "escalation_reason) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
        (datetime.now().isoformat(), query, category, severity, int(escalated),
         top_source, recommended_office, classifier_source, int(needs_review),
         escalation_reason),
    )
    conn.commit()
    interaction_id = cur.lastrowid
    conn.close()
    return interaction_id


def log_feedback(interaction_id: int, value: str):
    """Student clicked thumbs up/down on a fully-answered question. `value`
    is "up" or "down". This is the missing half of the assignment's
    "evaluate usefulness" requirement -- category/severity/escalation data
    alone can't tell you whether a generated answer was actually any good;
    only the student asking it can."""
    conn = _connect()
    conn.execute("UPDATE interactions SET feedback = ? WHERE id = ?", (value, interaction_id))
    conn.commit()
    conn.close()


def mark_resolved(interaction_id: int, note: str = ""):
    """Admin marks a needs-review item as handled. This is what makes the
    admin dashboard a real interface rather than a read-only report."""
    conn = _connect()
    conn.execute(
        "UPDATE interactions SET resolved = 1, admin_note = ? WHERE id = ?",
        (note, interaction_id),
    )
    conn.commit()
    conn.close()


def acknowledge(interaction_id: int, note: str = ""):
    """Admin has seen an escalated case and taken responsibility for it.
    Records WHEN, plus an optional note. Deliberately separate from
    `resolved` (which means a knowledge-base gap was fixed), so acknowledging
    a case never hides a needs-review item."""
    conn = _connect()
    conn.execute(
        "UPDATE interactions SET acknowledged = 1, acknowledged_at = ?, ack_note = ? WHERE id = ?",
        (datetime.now().isoformat(), (note or "").strip(), interaction_id),
    )
    conn.commit()
    conn.close()


def reopen(interaction_id: int):
    """Undo an acknowledgement (e.g. clicked by mistake): the case returns to
    the escalation queue and its acknowledgement time and note are cleared."""
    conn = _connect()
    conn.execute(
        "UPDATE interactions SET acknowledged = 0, acknowledged_at = NULL, ack_note = NULL WHERE id = ?",
        (interaction_id,),
    )
    conn.commit()
    conn.close()


def get_escalation_queue(df: pd.DataFrame) -> pd.DataFrame:
    """Escalated cases no admin has acknowledged yet, newest first."""
    if df.empty:
        return df
    ack = df["acknowledged"].fillna(0) if "acknowledged" in df else 0
    mask = (df["escalated"] == 1) & (ack == 0)
    return df[mask].sort_values(["timestamp", "id"], ascending=False)   # id breaks exact-time ties


def get_acknowledged(df: pd.DataFrame) -> pd.DataFrame:
    """Escalated cases an admin has acknowledged, most recently acknowledged first."""
    if df.empty or "acknowledged" not in df:
        return df.iloc[0:0]
    mask = (df["escalated"] == 1) & (df["acknowledged"].fillna(0) == 1)
    out = df[mask]
    if "acknowledged_at" in out:
        out = out.sort_values(["acknowledged_at", "id"], ascending=False, na_position="last")
    return out


def avg_minutes_to_acknowledge(df: pd.DataFrame) -> Optional[float]:
    """Mean minutes between a case being escalated and an admin acknowledging
    it, or None if no acknowledgement has a recorded time yet. This is the
    response-time figure a welfare office would actually track."""
    acked = get_acknowledged(df)
    if acked.empty or "acknowledged_at" not in acked:
        return None
    acked = acked.dropna(subset=["acknowledged_at"])
    if acked.empty:
        return None
    return float(((acked["acknowledged_at"] - acked["timestamp"]).dt.total_seconds() / 60).mean())


def get_unresolved(df: pd.DataFrame) -> pd.DataFrame:
    """Items flagged as needing admin attention (not in the knowledge base,
    or low-confidence retrieval) that haven't been marked resolved yet."""
    if df.empty:
        return df
    mask = (df.get("needs_review", 0) == 1) & (df.get("resolved", 0) == 0)
    return df[mask].sort_values("timestamp", ascending=False)


def load_all() -> pd.DataFrame:
    conn = _connect()
    df = pd.read_sql_query("SELECT * FROM interactions ORDER BY timestamp", conn)
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "acknowledged_at" in df:
            df["acknowledged_at"] = pd.to_datetime(df["acknowledged_at"], errors="coerce")
    return df


def category_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["category"].value_counts()
    return pd.DataFrame({"category": counts.index, "count": counts.values})


def escalation_reason_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Escalated cases grouped by WHY they were escalated (crisis,
    possible_crisis, gbv_report, high_severity, possible_high,
    low_confidence_sensitive...). Lets the dashboard show the queue isn't
    one undifferentiated pile -- a "crisis" case and a "low_confidence_sensitive"
    case are not the same kind of urgent."""
    if df.empty or "escalation_reason" not in df:
        return pd.DataFrame({"reason": [], "count": []})
    escalated = df[df["escalated"] == 1]
    reasons = escalated["escalation_reason"].replace("", "unspecified").fillna("unspecified")
    counts = reasons.value_counts()
    return pd.DataFrame({"reason": counts.index, "count": counts.values})


def daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.set_index("timestamp").resample("D").size().reset_index(name="queries")
    return daily


def feedback_summary(df: pd.DataFrame) -> dict:
    """Aggregate thumbs up/down counts and a helpfulness percentage for the
    admin dashboard. Interactions with no feedback given are excluded from
    the percentage entirely rather than counted as neutral or negative --
    most answers will never get a click either way, and folding silence
    into the rate would make it meaningless."""
    if df.empty or "feedback" not in df:
        return {"up": 0, "down": 0, "total": 0, "pct_helpful": None}
    fb = df["feedback"].dropna()
    fb = fb[fb != ""]
    up = int((fb == "up").sum())
    down = int((fb == "down").sum())
    total = up + down
    pct_helpful = (up / total * 100) if total else None
    return {"up": up, "down": down, "total": total, "pct_helpful": pct_helpful}


def escalation_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return df["escalated"].mean() * 100