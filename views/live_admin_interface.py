"""
Live Admin Interface: a real-time monitor of student interactions.

It reads the same database the chat writes to and refreshes itself, so:
open the User Interface in one browser tab, send a message, and watch it
arrive here in another tab within a few seconds.
"""
import html
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from ui_theme import wide

import analytics_db

# Blue "Acknowledge" buttons. Every button inside a form on this page is an
# Acknowledge button, so we target buttons inside Streamlit forms by several
# routes at once. !important beats Streamlit's and the theme's own button styles.
st.markdown(
    """
    <style>
    [data-testid="stForm"] button,
    form button,
    button[data-testid*="FormSubmit"] {
        background-color: #2563EB !important;
        border: 1px solid #2563EB !important;
        color: #ffffff !important;
    }
    [data-testid="stForm"] button p,
    form button p,
    button[data-testid*="FormSubmit"] p {
        color: #ffffff !important;
    }
    [data-testid="stForm"] button:hover,
    form button:hover,
    button[data-testid*="FormSubmit"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
    }
    [data-testid="stForm"] button:active,
    form button:active,
    button[data-testid*="FormSubmit"]:active {
        background-color: #1E40AF !important;
        border-color: #1E40AF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 🔴 Live Admin Interface")
st.caption("A real-time view of what students are sending right now. Open the **User "
           "Interface** in another browser tab, send a message, and it appears here "
           "within a few seconds with its category, severity and escalation status.")
st.info("🔒 Messages about self-harm or sexual violence are redacted before they are "
        "logged. Administrators see only their category, severity and escalation "
        "status, never the student's words.")

ctl1, ctl2, _ = st.columns([1.3, 1.6, 4])
live_on = ctl1.toggle("Auto-refresh", value=True, key="live_on")
interval = ctl2.selectbox("Refresh interval", [2, 3, 5, 10], index=1, key="live_interval",
                          format_func=lambda s: f"Every {s} seconds",
                          label_visibility="collapsed", disabled=not live_on)

REASONS = {
    "crisis": "Crisis message",
    "high_severity": "High severity",
    "gbv_report": "Harassment / GBV report",
    "possible_high": "Possible high severity (model)",
    "low_confidence_sensitive": "Low confidence on a sensitive topic",
    "possible_crisis": "Possible crisis wording (needs review)",
}
SEVERITIES = {"low", "medium", "high", "critical"}


def _reason(row) -> str:
    source = str(row.get("classifier_source") or "")
    if "escalated:" in source:
        code = source.split("escalated:")[-1].strip()
        return REASONS.get(code, code)
    if row["severity"] == "Critical":
        return REASONS["crisis"]
    return "Escalated"


def _ago(ts: pd.Timestamp, now: datetime) -> str:
    secs = max(0, int((now - ts).total_seconds()))
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60} min ago"
    return f"{secs // 3600} h ago"


def _fmt_minutes(m) -> str:
    if m is None:
        return "—"
    if m < 1:
        return "< 1 min"
    if m < 60:
        return f"{m:.0f} min"
    if m < 1440:
        return f"{m / 60:.1f} h"
    return f"{m / 1440:.1f} days"


def _ack(interaction_id: int) -> None:
    """Form callback: acknowledge with whatever note was typed, then clear the box."""
    key = f"ack_note_{interaction_id}"
    analytics_db.acknowledge(interaction_id, st.session_state.get(key, ""))
    st.session_state.pop(key, None)


def _feed_card(row, now: datetime) -> str:
    esc = html.escape
    sev = str(row["severity"])
    sev_cls = sev.lower() if sev.lower() in SEVERITIES else "low"
    is_esc = int(row["escalated"]) == 1
    acked = int(row.get("acknowledged") or 0) == 1
    fresh = (now - row["timestamp"]).total_seconds() <= 20

    tags = ""
    if is_esc:
        tags += f'<span class="badge badge-escalated">🚨 ESCALATED · {esc(_reason(row))}</span>'
    if acked:
        at = row.get("acknowledged_at")
        when = f" {at:%H:%M}" if pd.notna(at) else ""
        tags += f'<span class="badge badge-ack">✓ acknowledged{when}</span>'
    if fresh:
        tags += '<span class="badge badge-new">NEW</span>'

    office = str(row.get("recommended_office") or "")
    meta = f"→ {esc(office)}" if office else "no office referral logged"

    return (
        f'<div class="live-card{" escalated" if is_esc else ""}">'
        f'<div class="live-top">'
        f'<span class="live-time">{row["timestamp"]:%H:%M:%S} · {_ago(row["timestamp"], now)}</span>'
        f'<span class="badge badge-category">{esc(str(row["category"]))}</span>'
        f'<span class="badge badge-{sev_cls}">{esc(sev)} severity</span>{tags}</div>'
        f'<div class="live-text">{esc(str(row["query"]))}</div>'
        f'<div class="live-meta">{meta}</div></div>'
    )


def render_live() -> None:
    now = datetime.now()
    df = analytics_db.load_all()

    if df.empty:
        st.warning("No interactions yet. Open the **User Interface** in another tab and "
                   "send a message: it will show up here automatically.")
        return

    # Toast for escalations that arrived since the last refresh (none on first load).
    newest_id = int(df["id"].max())
    last_seen = st.session_state.get("live_last_seen_id")
    if last_seen is not None and newest_id > last_seen:
        arrived = df[(df["id"] > last_seen) & (df["escalated"] == 1)]
        for _, r in arrived.iterrows():
            st.toast(f"New escalated case: {r['category']} · {r['severity']}", icon="🚨")
    st.session_state["live_last_seen_id"] = newest_id

    queue = analytics_db.get_escalation_queue(df)
    last_5 = int((df["timestamp"] >= pd.Timestamp(now) - pd.Timedelta(minutes=5)).sum())

    acked = analytics_db.get_acknowledged(df)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total interactions", len(df))
    m2.metric("Open escalations", len(queue))
    m3.metric("Escalation rate", f"{analytics_db.escalation_rate(df):.1f}%")
    m4.metric("Avg time to acknowledge",
              _fmt_minutes(analytics_db.avg_minutes_to_acknowledge(df)))
    m5.metric("Last 5 minutes", last_5)
    st.caption(f"Last updated {now:%H:%M:%S}" + ("" if live_on else " (auto-refresh is off)"))

    st.divider()
    st.markdown("#### 🚨 Escalation queue: needs a person")
    if queue.empty:
        st.success("No open escalations. Every escalated case has been acknowledged.")
    else:
        for _, r in queue.head(8).iterrows():
            rid = int(r["id"])
            with st.container(border=True):
                st.markdown(f"**{r['category']}** · {r['severity']} severity · "
                            f"{r['timestamp']:%H:%M:%S} ({_ago(r['timestamp'], now)})")
                st.write(r["query"])
                st.caption(f"Why escalated: {_reason(r)}"
                           + (f" · Refer to: {r['recommended_office']}"
                              if r.get("recommended_office") else ""))
                # A form keeps a half-typed note safe while the page auto-refreshes.
                with st.form(key=f"ackform_{rid}", border=False):
                    note_col, btn_col = st.columns([4, 1])
                    note_col.text_input(
                        "Note", key=f"ack_note_{rid}", label_visibility="collapsed",
                        placeholder="Optional note, e.g. “Called student, referred to CEGENSA”")
                    btn_col.form_submit_button("Acknowledge", on_click=_ack, args=(rid,),
                                               **wide(st.form_submit_button))
        if len(queue) > 8:
            st.caption(f"…and {len(queue) - 8} more open escalations.")

    with st.expander(f"✓ Acknowledged cases ({len(acked)})"):
        if acked.empty:
            st.caption("Nothing acknowledged yet. Cases you acknowledge appear here, "
                       "with the time and any note, and can be reopened.")
        for _, r in acked.head(10).iterrows():
            rid = int(r["id"])
            with st.container(border=True):
                info, undo = st.columns([5, 1])
                with info:
                    st.markdown(
                        '<span style="background:#DCFCE7;color:#166534;border:1px solid #86EFAC;'
                        'border-radius:999px;padding:2px 10px;font-size:0.8rem;font-weight:600;">'
                        '✓ Acknowledged</span>',
                        unsafe_allow_html=True)
                    st.markdown(f"**{r['category']}** · {r['severity']} severity · "
                                f"raised {r['timestamp']:%H:%M:%S}")
                    st.write(r["query"])
                    at = r.get("acknowledged_at")
                    if pd.notna(at):
                        mins = (at - r["timestamp"]).total_seconds() / 60
                        line = f"Acknowledged {at:%H:%M:%S} ({_fmt_minutes(mins)} after it was raised)"
                    else:
                        line = "Acknowledged (time not recorded)"
                    note = str(r.get("ack_note") or "").strip()
                    st.caption(line + (f" · Note: {note}" if note else ""))
                with undo:
                    st.button("Reopen", key=f"reopen_{rid}", on_click=analytics_db.reopen,
                              args=(rid,), **wide(st.button))

    st.divider()
    feed_col, chart_col = st.columns([3, 2])
    with feed_col:
        st.markdown("#### Live feed (newest first)")
        recent = df.sort_values("timestamp", ascending=False).head(15)
        st.markdown("".join(_feed_card(r, now) for _, r in recent.iterrows()),
                    unsafe_allow_html=True)
    with chart_col:
        st.markdown("#### Demand by category")
        counts = analytics_db.category_counts(df)
        fig = px.bar(counts, x="count", y="category", orientation="h", color="category")
        fig.update_layout(showlegend=False, xaxis_title="Queries", yaxis_title="",
                          margin=dict(l=0, r=0, t=10, b=0), height=300)
        st.plotly_chart(fig, **wide(st.plotly_chart), key="live_cat_chart")

        st.markdown("#### Severity mix")
        sev = df["severity"].value_counts()
        fig2 = px.pie(pd.DataFrame({"severity": sev.index, "count": sev.values}),
                      names="severity", values="count", hole=0.45)
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=260)
        st.plotly_chart(fig2, **wide(st.plotly_chart), key="live_sev_chart")


# A fragment re-runs on its own timer, so only this block refreshes (not the
# whole page), and the toggle above simply switches the timer on or off.
st.fragment(render_live, run_every=(interval if live_on else None))()