import os
from datetime import datetime

import streamlit as st

import config
import neural_classifier
import risk_classifier

st.markdown("## ⚙️ Settings")
st.caption("System status, the indexed knowledge base, and a live classifier test tool — "
           "kept here so the User Interface page stays focused on the conversation.")

# --- System status -------------------------------------------------------
st.divider()
st.markdown("#### System status")

llm_ok = config.llm_is_configured()
clf_ok = neural_classifier.is_available()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("LLM provider", config.LLM_PROVIDER)
    if config.LLM_PROVIDER == "ollama":
        st.caption(f"Model: `{config.OLLAMA_MODEL}` @ {config.OLLAMA_HOST}")
    st.caption("Connected" if llm_ok else "Template mode — no API key/local server configured")
with col2:
    st.metric("Classifier", "Trained" if clf_ok else "Fallback")
    if clf_ok and os.path.exists(config.CLASSIFIER_MODEL_PATH):
        trained_at = datetime.fromtimestamp(os.path.getmtime(config.CLASSIFIER_MODEL_PATH))
        st.caption(f"Neural network — last trained {trained_at:%Y-%m-%d %H:%M}")
    else:
        st.caption("Rule-based keywords — run train_classifier.py")
with col3:
    st.metric("Retrieval", "Local")
    st.caption(f"Embedding model: `{config.EMBEDDING_MODEL_NAME}` — no external API calls")

st.caption(f"Retrieval confidence threshold: `1.0` (set in `views/user_interface.py` — above this distance, the assistant "
           f"says it doesn't know rather than guessing)")

# --- Privacy notice --------------------------------------------------------
st.warning("⚠️ **Prototype only — not production access control.** No authentication in front of "
           "this app or the Admin Interface. Crisis-flagged messages are redacted at the point of "
           "logging, but other query text is stored in plaintext with no retention policy. See "
           "`report_support_material.md` for the full write-up.")

# --- Knowledge base --------------------------------------------------------
st.divider()
st.markdown("#### Indexed knowledge base")
st.caption(f"{len(config.MARKDOWN_FILES)} source documents currently indexed. Sizes and "
           f"last-modified dates below help spot a source that looks stale or didn't index.")
for f in config.MARKDOWN_FILES:
    if os.path.exists(f):
        size_kb = os.path.getsize(f) / 1024
        modified = datetime.fromtimestamp(os.path.getmtime(f))
        st.markdown(f"- `{f}` — {size_kb:.1f} KB, updated {modified:%Y-%m-%d}")
    else:
        st.markdown(f"- `{f}` — ⚠️ **file not found**")

st.divider()
st.markdown("#### Categories the system routes to")
for c in config.CATEGORIES:
    st.markdown(f"- {c}")

# --- Live classifier test tool ---------------------------------------------
st.divider()
st.markdown("#### Test the classifier")
st.caption("Type a sample question to see exactly how it would be categorized — without going "
           "through the full chat flow. Useful for checking the classifier's behavior directly.")

test_query = st.text_input("Sample question", placeholder="e.g. I can't afford my hostel fees this semester")
if test_query:
    normalized = risk_classifier.normalize_query(test_query)
    if normalized != test_query:
        st.caption(f"Typo-corrected to: *{normalized}*")

    is_crisis = risk_classifier.check_crisis(normalized)
    is_smalltalk = risk_classifier.is_chitchat(normalized)

    if is_crisis:
        st.error("🚨 Would be flagged as a **crisis message** — routed straight to emergency "
                 "referral, bypassing classification and the LLM entirely.")
    elif is_smalltalk:
        st.info("💬 Would be treated as **chit-chat/greeting** — gets a natural reply, not "
                "forced through classification.")
    else:
        result_col1, result_col2 = st.columns(2)
        net_result = None
        with result_col1:
            st.markdown("**Neural classifier**")
            if neural_classifier.is_available():
                category, severity, confidence, p_high = neural_classifier.classify_full(normalized)
                net_result = (category, severity, confidence, p_high)
                st.write(f"Category: **{category}**")
                st.write(f"Severity: **{severity}**")
                st.write(f"Confidence: **{confidence:.0%}**")
                if p_high is not None:
                    st.write(f"P(High severity): **{p_high:.0%}**")
            else:
                st.caption("Not trained yet — run train_classifier.py")
        with result_col2:
            st.markdown("**Rule-based fallback**")
            triage = risk_classifier.rule_based_classify(normalized)
            st.write(f"Category: **{triage.category}**")
            st.write(f"Severity: **{triage.severity}**")
            st.write(f"Matched rule: `{triage.matched_rule}`")

        # What the live app would actually do: net (or rules) + safety net +
        # urgency floor + escalation policy.
        st.markdown("**Final decision (after safety net, urgency floor and escalation policy)**")
        if net_result is not None:
            base_cat, base_sev, base_conf, base_ph = net_result
        else:
            base_cat, base_sev, base_conf, base_ph = triage.category, triage.severity, None, None
        f_cat, f_sev, f_esc, f_reason, f_note = risk_classifier.finalize_triage(
            normalized, base_cat, base_sev, base_conf, base_ph)
        st.write(f"Category: **{f_cat}**  ·  Severity: **{f_sev}**  ·  "
                 f"Escalated: **{'YES — ' + f_reason if f_esc else 'no'}**")
        if f_note:
            st.caption(f"Adjustments applied: {f_note}")


# --- Generation & decision tuning -----------------------------------------
import streamlit as st
st.divider()
st.markdown("#### Generation & decision tuning")
st.caption(
    "Session-only � resets when you close the tab. "
    "Classification and chit-chat replies are excluded so eval numbers stay reproducible."
)

col_a, col_b = st.columns(2)

with col_a:
    st.session_state["tuning_temperature"] = st.slider(
        "LLM temperature",
        min_value=0.0, max_value=1.0,
        value=st.session_state.get("tuning_temperature", 0.3),
        step=0.05,
        help="Higher = more creative answers. Lower = more consistent and factual.",
    )
    st.session_state["tuning_classifier_cutoff"] = st.slider(
        "Classifier confidence cutoff",
        min_value=0.30, max_value=0.95,
        value=st.session_state.get("tuning_classifier_cutoff", 0.55),
        step=0.05,
        help="Below this confidence the neural classifier defers to rule-based fallback.",
    )

with col_b:
    st.session_state["tuning_max_tokens"] = st.slider(
        "Max response tokens",
        min_value=100, max_value=1000,
        value=st.session_state.get("tuning_max_tokens", 500),
        step=50,
        help="Maximum tokens the LLM can generate per answer.",
    )
    st.session_state["tuning_retrieval_cutoff"] = st.slider(
        "Retrieval confidence cutoff",
        min_value=0.50, max_value=1.50,
        value=st.session_state.get("tuning_retrieval_cutoff", 1.0),
        step=0.05,
        help="ChromaDB distance threshold � above this the assistant says it does not know.",
    )
