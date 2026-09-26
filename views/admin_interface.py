import streamlit as st

from ui_theme import wide
import plotly.express as px
import pandas as pd
import os

import analytics_db
import topic_modeling

# --- Minimal password gate ---------------------------------------------
# The warning below was true when this said "no authentication" -- this is
# the fix for that specific gap before any public deployment. It's still a
# single shared password, not per-admin accounts or roles, so it belongs in
# the same "prototype, not production access control" category as the rest
# of this page: fine for keeping random visitors and search-engine crawlers
# out of a demo link you're sharing with a supervisor, not a substitute for
# real authentication (individual logins, audit trail, role separation) in
# an actual production deployment handling real student welfare data.
#
# Set ADMIN_PASSWORD in Streamlit Cloud's Secrets panel (or a local .env /
# environment variable) before deploying. If it's never set, this page
# refuses to render at all rather than silently falling open -- an admin
# page with no password configured is not the same as "no password needed".
_ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
if not _ADMIN_PASSWORD:
    try:
        _ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", None)
    except Exception:
        # No secrets.toml at all (normal for local dev without one) --
        # treat the same as "not configured", handled below.
        _ADMIN_PASSWORD = None

if not _ADMIN_PASSWORD:
    st.error("Admin access is not configured (ADMIN_PASSWORD is not set). This page is "
             "locked until an administrator sets it in the app's secrets/environment.")
    st.stop()

if not st.session_state.get("admin_authenticated", False):
    st.markdown("## 🔒 Admin sign-in")
    entered = st.text_input("Admin password", type="password")
    if st.button("Sign in"):
        if entered == _ADMIN_PASSWORD:
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

st.markdown("## 📊 Admin Interface")
st.caption("This is the decision-support view for administrators: what students are asking about, "
           "how demand moves over time, and how often cases needed escalation.")
st.warning("⚠️ **Prototype only — not production access control.** A shared password gates this "
           "page now, but that's still not real per-admin authentication, an audit trail, or role "
           "separation. Crisis-flagged and sexual-harassment/GBV messages are redacted at the "
           "point of logging (see views/user_interface.py), but non-crisis query text is still "
           "stored in plaintext. A real deployment needs: individual authenticated admin accounts, "
           "a data retention/deletion policy, and a documented legal basis for storing student "
           "welfare data.")

df = analytics_db.load_all()

if df.empty:
    st.info("No interactions logged yet. Ask the assistant a few questions on the User Interface page first.")
    st.stop()

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total interactions", len(df))
col2.metric("Escalation rate", f"{analytics_db.escalation_rate(df):.1f}%")
col3.metric("Distinct categories", df["category"].nunique())
ack_count = int(((df["escalated"] == 1) & (df.get("acknowledged", 0).fillna(0) == 1)).sum()) if "acknowledged" in df else 0
col4.metric("Acknowledged escalations", ack_count)
avg_ack = analytics_db.avg_minutes_to_acknowledge(df)
col5.metric("Avg. acknowledgement", "—" if avg_ack is None else f"{avg_ack:.1f} min")
fb = analytics_db.feedback_summary(df)
col6.metric("Helpfulness",
            "—" if fb["pct_helpful"] is None else f"{fb['pct_helpful']:.0f}%",
            help=f"{fb['up']} up · {fb['down']} down · {fb['total']} of {len(df)} "
                 f"interactions rated" if fb["total"] else "No feedback given yet")

st.divider()

left, right = st.columns(2)

with left:
    st.markdown("#### Issue category demand")
    counts = analytics_db.category_counts(df)
    fig = px.bar(counts, x="category", y="count", color="category")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Queries")
    st.plotly_chart(fig, **wide(st.plotly_chart))

with right:
    st.markdown("#### Severity breakdown")
    sev_series = df["severity"].value_counts()
    sev_counts = pd.DataFrame({"severity": sev_series.index, "count": sev_series.values})
    fig2 = px.pie(sev_counts, names="severity", values="count", hole=0.4)
    st.plotly_chart(fig2, **wide(st.plotly_chart))

st.markdown("#### Daily query trend")
trend = analytics_db.daily_trend(df)
fig3 = px.line(trend, x="timestamp", y="queries", markers=True)
fig3.update_layout(xaxis_title="Date", yaxis_title="Queries")
st.plotly_chart(fig3, **wide(st.plotly_chart))

st.divider()
st.markdown("#### Referral distribution — which offices are getting the traffic")
referred = df[df["recommended_office"].fillna("") != ""]
if referred.empty:
    st.caption("No referrals logged yet.")
else:
    office_counts = referred["recommended_office"].value_counts().reset_index()
    office_counts.columns = ["office", "count"]
    fig4 = px.bar(office_counts, x="count", y="office", orientation="h")
    fig4.update_layout(yaxis_title="", xaxis_title="Referrals")
    st.plotly_chart(fig4, **wide(st.plotly_chart))

st.divider()
st.markdown("#### Recent high-severity / escalated cases")
flagged = df[df["escalated"] == 1].sort_values("timestamp", ascending=False)
if flagged.empty:
    st.caption("No escalations logged yet.")
else:
    cols = [c for c in ["timestamp", "query", "category", "severity", "acknowledged",
                        "acknowledged_at", "ack_note"] if c in flagged.columns]
    st.dataframe(flagged[cols], **wide(st.dataframe))

st.divider()
st.markdown("#### Recurring themes in unanswered questions")
st.caption("Unsupervised topic discovery over the 'needs review' queue below -- run on demand "
           "since it's real computation, not just a chart of logged numbers. This looks for "
           "patterns without being told what to look for, so it can surface a cluster of related "
           "questions even when they don't fit neatly into one of the fixed categories above.")
if st.button("Find recurring themes"):
    unresolved_for_topics = analytics_db.get_unresolved(df)
    try:
        result = topic_modeling.discover_topics(unresolved_for_topics["query"].tolist())
    except Exception as e:
        result = {"ok": False, "reason": f"Topic modeling failed unexpectedly: {e}"}
    if not result["ok"]:
        st.info(result["reason"])
    else:
        texts_list = unresolved_for_topics["query"].tolist()
        for topic in result["topics"]:
            with st.container(border=True):
                st.markdown("**" + " · ".join(topic["keywords"]) + "**")
                st.caption(f"{len(topic['doc_indices'])} question(s) most closely match this theme")
                for i in topic["doc_indices"][:5]:
                    st.write(f"— {texts_list[i]}")

st.divider()
st.markdown("#### Needs review — questions not in the knowledge base")
st.caption("These are real gaps: questions the system couldn't confidently answer from what's "
           "indexed. Mark one resolved once you've added the relevant content (or confirmed it's "
           "genuinely out of scope) — this is what makes this a working interface rather than a "
           "read-only report.")
unresolved = analytics_db.get_unresolved(df)
if unresolved.empty:
    st.caption("Nothing pending — every flagged gap has been addressed.")
else:
    for _, row in unresolved.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**{row['category']}** · {row['timestamp']}")
                st.write(row["query"])
            with c2:
                if st.button("Mark resolved", key=f"resolve_{row['id']}"):
                    analytics_db.mark_resolved(int(row["id"]))
                    st.rerun()

st.divider()
with st.expander("Raw interaction log"):
    st.dataframe(df, **wide(st.dataframe))