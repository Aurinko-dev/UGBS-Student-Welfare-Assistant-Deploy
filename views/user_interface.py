"""User Interface: the student-facing chat (selected from the sidebar in app.py)."""
import streamlit as st

from logo import logo_html
from ui_theme import wide
import os

from langchain_chroma import Chroma

import config
import analytics_db
import llm_engine
import neural_classifier
import action_planner
import program_guidance
from risk_classifier import (rule_based_classify, check_crisis, is_chitchat,
                             normalize_query, finalize_triage)

# --- Retrieval confidence threshold ---------------------------------------
# Chroma's default distance is L2 on normalized embeddings; empirically for
# all-MiniLM-L6-v2 a distance above ~0.9 means "not actually a good match".
# Below this we tell the student we don't know rather than guessing -- but
# we still give them a direct website link and log it for admin follow-up,
# rather than a dead end (see llm_engine.get_not_in_kb_reply).
CONFIDENCE_THRESHOLD = 1.0
# Raised from 0.9 to 1.0 after running check_retrieval_scores.py on real
# queries from the live app. At 0.9, several genuinely good matches were
# being rejected (observed distances: 0.91 for "who is my course advisor",
# 0.93 for a short follow-up on the Analytics option, 0.94 for "what
# options can I choose at UGBS"), while clearly bad matches sit at 1.22+
# and are still correctly rejected at 1.0. This is not a final answer --
# it is what the numbers from ONE run on ONE machine support. Two things
# raising this threshold does NOT fix, and which raising it further would
# only make worse: (1) some in-range matches are still the WRONG document
# for the question (e.g. "book a counselling appointment" matches the
# Career Counselling FAQ, not the UGCCD personal-counselling policy, at
# distance 0.66 -- well inside any reasonable threshold); (2) some topics
# (roommate conflicts, academic probation from the student's point of
# view) have no good match in the knowledge base at any threshold, because
# the content doesn't exist yet. Re-run check_retrieval_scores.py and
# re-tune this number whenever documents are added, removed, or edited.

# Simple keyword check for "is this question actually about a specific fee
# amount" -- used to decide whether to ask nationality before answering.
# Deliberately narrow: a general accommodation/financial question that
# ISN'T about a number doesn't need this extra step.
_FEE_SPECIFIC_HINTS = ["fee", "fees", "pay", "cost", "how much", "tuition", "price"]

# --- Course-advisor department gate ----------------------------------------
# "Who is my course advisor?" has no single answer -- it depends on which
# option/department the student is in, and the assistant has no per-student
# record to look that up. Rather than a generic or "I don't know" answer,
# ask which department first (same pattern as the nationality gate below),
# then answer using ugbs_offices_and_contacts.md's per-department entry.
_ADVISOR_HINTS = ["course advisor", "course adviser", "my advisor", "my adviser",
                 "who is my advisor", "who is my adviser"]

DEPARTMENTS = {
    "Accounting": "Accounting (including the Diploma in Accounting)",
    "Banking and Finance / Insurance": "Finance",
    "Marketing / E-Commerce and Customer Management": "Marketing and Entrepreneurship",
    "Human Resource Management": "Organisation and Human Resource Management",
    "Analytics": "Operations and Management Information Systems (OMIS)",
    "Health Services Management": "Health Services Management",
    "Public Administration": "Public Administration",
}


def _is_advisor_question(text: str) -> bool:
    lowered = text.lower()
    return any(h in lowered for h in _ADVISOR_HINTS)

# Shown when the knowledge base has no confident answer. The chat is anonymous,
# so an admin cannot reply here -- the wording says that honestly rather than
# promising a callback.
_ADMIN_HANDOFF = (
    "\n\n---\n"
    "**I've flagged your question for the UGBS admin team** so the gap can be "
    "reviewed and the assistant improved. They can't reply in this chat, so for "
    "a direct answer please contact the relevant office or visit "
    f"{config.UGBS_WEBSITE}."
)


def _render_feedback(interaction_id: int) -> None:
    """Thumbs up/down under a fully-answered question. Once given, feedback
    is locked in as plain text rather than left clickable -- so a student
    can't repeatedly toggle it, and so the buttons don't reappear as live
    on every later rerun of this page."""
    given = st.session_state.feedback_given.get(interaction_id)
    if given:
        st.caption("👍 Marked helpful" if given == "up" else "😞 Marked not helpful")
        return
    c1, c2, _ = st.columns([1, 1, 10])
    if c1.button("👍", key=f"fb_up_{interaction_id}"):
        analytics_db.log_feedback(interaction_id, "up")
        st.session_state.feedback_given[interaction_id] = "up"
        st.rerun()
    if c2.button("😞", key=f"fb_down_{interaction_id}"):
        analytics_db.log_feedback(interaction_id, "down")
        st.session_state.feedback_given[interaction_id] = "down"
        st.rerun()


def _loggable(query: str, category: str) -> str:
    """Same redaction policy as the crisis path: sexual harassment / GBV
    disclosures are never stored verbatim in the dashboard-visible log."""
    if category == "Sexual Harassment / GBV":
        return "[GBV message — redacted from log]"
    return query


EMERGENCY_MESSAGE_GBV = """### 🚨 This sounds serious — you deserve support from a real person

I'm an AI intake assistant, so I can't handle this on my own. Please reach out
to someone who can help right now:

- **CEGENSA (Centre for Gender Studies and Advocacy)** — runs a dedicated
  sexual harassment crisis and counselling unit for students: https://cegensa.ug.edu.gh
- **University of Ghana Medical Centre** — has an emergency department, and is the
  fastest route if you are hurt or in immediate danger.
- **The Police** — for serious incidents such as rape or assault, the University's
  policy advises reporting to the Police as well as to the Anti-Sexual Harassment
  Committee.
- **Careers and Counselling Directorate (UGCCD)** — confidential emotional support.
- **A trusted friend, family member, or hall/hostel warden** — please tell someone
  near you what is going on.

You do not have to file a formal complaint to get support, and you will not be
penalised for reporting in good faith. What happened is not your fault.
"""

EMERGENCY_MESSAGE = """### 🚨 This sounds urgent — please don't wait on this chat

I'm an AI intake assistant, and I'm not able to help with a crisis on my own.
Please reach out to a real person right now:

- **University of Ghana Counselling and Placement Centre (UGCCD)** — go to the
  Centre in person if you can, or contact them directly.
- **University of Ghana Medical Centre** — has an emergency department and is
  the fastest route if you or someone else is in immediate danger.
- **A trusted friend, family member, or hall/hostel warden** — please tell
  someone near you what's going on right now.

You matter, and this is worth a real person's attention, not a chatbot's.
"""

import os as _os, build_vectorstore as _bvs
if not _os.path.exists(config.DB_DIR):
    _bvs.build_and_save_vectorstore(_bvs.chunk_documents(_bvs.load_md_documents()))


def _db_fingerprint_now():
    """Changes whenever the committed vector-DB files change (new commit or
    rebuild): built from every file's relative path + size. The segment folder
    is named by a fresh UUID on each rebuild, so the fingerprint changes too."""
    parts = []
    for root, _dirs, files in os.walk(config.DB_DIR):
        for f in files:
            p = os.path.join(root, f)
            try:
                parts.append(f"{os.path.relpath(p, config.DB_DIR)}:{os.path.getsize(p)}")
            except OSError:
                pass
    return "|".join(sorted(parts))


@st.cache_resource(show_spinner=False)
def load_vectorstore(db_fingerprint=None):
    # IMPORTANT: no leading underscore on db_fingerprint. Streamlit does NOT
    # hash underscore-prefixed arguments, which would make the cache key
    # constant and the cache-busting a no-op.
    embeddings = neural_classifier.get_embedder()
    if not os.path.exists(config.DB_DIR):
        return None
    return Chroma(persist_directory=config.DB_DIR, embedding_function=embeddings)


@st.cache_resource(show_spinner=False)
def _build_memory_vectorstore():
    """Last resort if the committed DB can't be used: index the markdown
    knowledge base in memory (slow on the first question only)."""
    import build_vectorstore
    docs = build_vectorstore.load_md_documents()
    chunks = build_vectorstore.chunk_documents(docs)
    return Chroma.from_documents(documents=chunks, embedding=neural_classifier.get_embedder())


def _search_with_recovery(vs, query, k=3):
    """Search, and if the cached Chroma client is stale/broken: clear caches,
    reopen the on-disk DB and retry; if that also fails, use an in-memory index."""
    try:
        return vs.similarity_search_with_score(query, k=k)
    except Exception as first_err:
        print(f"[user_interface] vector search failed ({type(first_err).__name__}: {first_err}); reloading DB")
    try:
        load_vectorstore.clear()
        try:
            from chromadb.api.shared_system_client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        except Exception as e:
            print(f"[user_interface] could not clear chroma system cache: {e}")
        fresh = load_vectorstore(_db_fingerprint_now())
        return fresh.similarity_search_with_score(query, k=k)
    except Exception as second_err:
        print(f"[user_interface] reload failed ({type(second_err).__name__}: {second_err}); using in-memory index")
    return _build_memory_vectorstore().similarity_search_with_score(query, k=k)


def severity_badge(severity: str) -> str:
    # Internal/admin use only (e.g. analytics dashboard) -- deliberately NOT
    # shown to the student. Surfacing a raw "Critical severity" / "High
    # severity" label risks two failure modes: (1) it reads as a diagnosis
    # of the student's situation, which this system has no business making,
    # and (2) it can teach people to soften how they phrase a disclosure to
    # avoid the label, which is the opposite of what a welfare tool wants.
    # The severity still fully drives escalation and routing behind the
    # scenes -- it's just never printed at the student.
    cls = f"badge-{severity.lower()}"
    return f'<span class="badge {cls}">{severity} severity</span>'


ASSISTANT_AVATAR = "assets/avatar_assistant.png"
USER_AVATAR = "assets/avatar_user.png"


# --- Topic boxes: each opens a few example questions ---------------------
TOPICS = {
    "Financial": {
        "icon": "💰",
        "questions": [
            "What are the eligibility requirements for SFAO financial aid?",
            "How do I apply for a scholarship?",
            "I can't pay my fees this semester. What can I do?",
            "Is there a payment plan for tuition fees?",
        ],
    },
    "Academic": {
        "icon": "📚",
        "questions": [
            "What is the policy on probation and academic standing?",
            "How do I apply to defer my programme?",
            "What happens if I fail a course and need to resit?",
            "How do I register my courses?",
        ],
    },
    "Graduation": {
        "icon": "📜",
        "questions": [
            "How do I know if I have satisfied all the requirements for graduation?",
            "How do I know I am eligible to graduate?",
            "How many credit hours do I need to pass to be eligible for graduation?",
            "How many total credit hours do I need to take to be eligible for graduation?",
            "Can I graduate with an F in an elective course under the College of Humanities?",
            "Can I graduate with an E in a core course in the College of Humanities?",
            "I satisfied all requirements but my name is not on the graduating list.",
            "I have written all my re-sit courses. How can I be added to the graduation list?",
            "I completed my course of study last year with some outstanding results. They are now entered but my name isn't on this year's graduating list.",
            "Can I be part of the graduation ceremony after completing school when I have passed all re-sit papers?",
            "What should I do when my name does not appear correctly for the online registration for graduation?",
            "If I fail to participate in matriculation and fail to sign the matriculation oath, will I be allowed to graduate?",
        ],
    },
    "Accommodation": {
        "icon": "🏠",
        "questions": [
            "How does the random bed allocation process work?",
            "How do I get accommodation on campus?",
            "What can I do about a problem with my roommate?",
            "I paid my hostel fees but haven't been allocated a room yet.",
        ],
    },
    "Programmes": {
        "icon": "🎓",
        "guidance": True,
        "questions": [
            "What options can I choose at UGBS?",
            "What does the Analytics option involve?",
            "Which department runs the Marketing option?",
            "Who is my course advisor?",
        ],
    },
    "Personal": {
        "icon": "💙",
        "questions": [
            "How can I book an appointment with a counsellor at UGCCD?",
            "Is counselling confidential?",
            "I've been feeling overwhelmed and anxious about school.",
            "Where can I report sexual harassment?",
        ],
    },
}

if "active_topic" not in st.session_state:
    st.session_state.active_topic = None

# An example question chosen on the previous run is picked up here, so the
# topic panel can close before the answer is shown.
suggested_prompt = st.session_state.pop("pending_prompt", None)

st.markdown(
    '<div class="ugbs-header">'
    + logo_html(64) +
    '<div><div class="main-header">UGBS Student Welfare Assistant</div>'
    '<div class="sub-header">Welcome to the UGBS Student Welfare Assistant. '
    'How may I help you today?</div></div></div>'
    '<div class="ugbs-rule"></div>',
    unsafe_allow_html=True)

st.markdown(
    '<p class="assist-prompt">How can I assist you?</p>'
    '<p class="assist-hint">Pick a topic below to see common questions, or simply '
    'type your own in the chat box at the bottom of the page.</p>',
    unsafe_allow_html=True)

topic_cols = st.columns(len(TOPICS))
for col, (topic_name, topic) in zip(topic_cols, TOPICS.items()):
    with col:
        is_open = st.session_state.active_topic == topic_name
        if st.button(f"{topic['icon']}  {topic_name}", key=f"topic_{topic_name}",
                     type="primary" if is_open else "secondary",
                     **wide(st.button)):
            st.session_state.active_topic = None if is_open else topic_name
            st.session_state.show_guidance = False  # always start collapsed
            st.rerun()

active_topic = st.session_state.active_topic
if active_topic:
    try:
        examples_box = st.container(key="examples")
    except TypeError:            # very old Streamlit: no container keys
        examples_box = st.container()
    with examples_box:
        st.caption(f"Common {active_topic.lower()} questions. Click one to ask it, "
                   f"or click **{active_topic}** again to close.")
        for i, question in enumerate(TOPICS[active_topic]["questions"]):
            if st.button(question, key=f"ex_{active_topic}_{i}", **wide(st.button)):
                st.session_state.pending_prompt = question
                st.session_state.active_topic = None
                st.session_state.show_guidance = False
                st.rerun()

        if TOPICS[active_topic].get("guidance"):
            # The quiz is its own button in the list, not something that
            # renders automatically just because the Programmes panel is
            # open -- matches every other entry here being a click-to-open
            # item, not an always-visible block.
            is_guidance_open = st.session_state.get("show_guidance", False)
            if st.button("🧠­ What major might suit me? (quick quiz)",
                        key=f"guidance_toggle_{active_topic}", **wide(st.button)):
                st.session_state.show_guidance = not is_guidance_open
                st.rerun()
            if st.session_state.get("show_guidance", False):
                st.divider()
                tab_ugbs, tab_all = st.tabs(["UGBS options (Level 200)", "All majors (Humanities Handbook)"])
                with tab_ugbs:
                    program_guidance.render("ugbs")
                with tab_all:
                    program_guidance.render("all")


_db_fingerprint = _db_fingerprint_now() if os.path.exists(config.DB_DIR) else None
vectorstore = load_vectorstore(_db_fingerprint)
if vectorstore is None:
    st.error("Local database `ugbs_welfare_db` not found. Please run `python build_vectorstore.py` first.")
    st.stop()

# The chat starts empty: the welcome message above replaces the old greeting bubble.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "nationality" not in st.session_state:
    st.session_state.nationality = None
if "awaiting_nationality" not in st.session_state:
    st.session_state.awaiting_nationality = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "awaiting_department" not in st.session_state:
    st.session_state.awaiting_department = False
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}

for msg in st.session_state.messages:
    avatar = ASSISTANT_AVATAR if msg["role"] == "assistant" else USER_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        # unsafe_allow_html deliberately NOT used here. Every message in this
        # history is either a student's raw typed input, or an assistant
        # reply built from that input plus retrieved/LLM-generated text --
        # neither is fully trusted. Rendering it as real HTML on every rerun
        # would let a message containing e.g. <img src=x onerror=...> execute
        # script in this session every time the page reruns, even though the
        # very same content is shown safely (no HTML execution) the first
        # time it appears further down in this file. Plain Markdown
        # (headers, bold, links, lists) still renders correctly without this
        # flag -- only raw HTML tags need it, and nothing here should be
        # emitting those.
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("interaction_id") is not None:
            _render_feedback(msg["interaction_id"])

user_query = st.chat_input("Type your question here...") or suggested_prompt

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        # --- Stage 0: crisis check, ALWAYS first, before any pending gate --
        # A student could be replying to "are you Ghanaian or international?"
        # -- or they could have sent something unrelated while that gate was
        # still open (a stray "hi", a new question, or worse, a crisis
        # message). The gates below must never be able to swallow a crisis
        # or a greeting, so those checks run first, unconditionally, and
        # clear any stuck gate before doing anything else.
        if check_crisis(user_query):
            st.session_state.awaiting_nationality = False
            st.session_state.pending_query = None
            st.session_state.awaiting_department = False
            crisis_category = rule_based_classify(user_query).category
            crisis_message = (EMERGENCY_MESSAGE_GBV
                              if crisis_category == "Sexual Harassment / GBV"
                              else EMERGENCY_MESSAGE)
            st.markdown(crisis_message)  # pure Markdown; no HTML needed, see history-loop note above
            analytics_db.log_interaction("[crisis message — redacted from log]",
                                          crisis_category, "Critical",
                                          escalated=True)
            st.session_state.messages.append({"role": "assistant", "content": crisis_message})
            st.stop()

        # --- Stage 0.2: chit-chat, also before any pending gate -------------
        # A stray "hi" sent while a gate was open must get a normal greeting
        # reply, not be silently absorbed as the answer to a stale question.
        if is_chitchat(user_query):
            st.session_state.awaiting_nationality = False
            st.session_state.pending_query = None
            st.session_state.awaiting_department = False
            reply = llm_engine.generate_chitchat_reply(user_query)
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.stop()

        # --- Stage 0.5: resuming a paused nationality question ----------------
        # If we asked "are you Ghanaian or international?" last turn, and the
        # message wasn't a crisis or a greeting (checked above), treat it as
        # the answer and resume processing the ORIGINAL question, not this one.
        if st.session_state.awaiting_nationality:
            lowered_answer = user_query.lower()
            if "ghana" in lowered_answer:
                st.session_state.nationality = "Ghanaian"
            elif any(w in lowered_answer for w in ("international", "foreign", "non-ghanaian", "not ghanaian")):
                st.session_state.nationality = "International"
            else:
                # Didn't parse cleanly -- don't block the student, just
                # carry their raw answer through as context.
                st.session_state.nationality = user_query.strip()
            st.session_state.awaiting_nationality = False
            user_query = st.session_state.pending_query
            st.session_state.pending_query = None

        # --- Stage 0.7: resuming a paused course-advisor department question -
        # If we asked "which department are you in?" last turn, and the
        # message wasn't a crisis or a greeting (checked above), treat it as
        # the department, not a new question -- answer directly and stop,
        # rather than running it through crisis/classification/retrieval as
        # if it were a welfare query.
        if st.session_state.awaiting_department:
            st.session_state.awaiting_department = False
            picked_label = user_query.strip()
            matched = next((full for label, full in DEPARTMENTS.items()
                           if picked_label.lower() in label.lower()
                           or label.lower() in picked_label.lower()), None)
            if matched is None:
                msg = ("I didn't recognise that department. Could you pick one from "
                       "the list above, or tell me your option/major (e.g. Accounting, "
                       "Marketing, Analytics)?")
                st.session_state.awaiting_department = True  # ask again, don't drop it
            else:
                msg = (f"For **{matched}**, ask your department office who your current "
                       f"course advisor is -- the assistant doesn't hold a live directory "
                       f"of individual advisors, only which department to contact.")
            st.markdown(msg)  # pure Markdown; no HTML needed, see history-loop note above
            analytics_db.log_interaction(f"[Course advisor] {picked_label}", "Career Guidance",
                                         "Low", escalated=False,
                                         classifier_source="advisor_department_gate")
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.stop()

        # --- Stage 1: deterministic crisis check (again) --------------------
        # Reaching here means the message was neither crisis nor chit-chat,
        # AND wasn't consumed by a pending gate above (or WAS a resumed
        # original question after the nationality gate). This second check
        # catches that resumed original question, in case IT happens to be
        # a crisis message -- unlikely, but the same "never let a crisis
        # message skip the crisis screen" rule applies here too.
        if check_crisis(user_query):
            crisis_category = rule_based_classify(user_query).category
            crisis_message = (EMERGENCY_MESSAGE_GBV
                              if crisis_category == "Sexual Harassment / GBV"
                              else EMERGENCY_MESSAGE)
            st.markdown(crisis_message)  # pure Markdown; no HTML needed, see history-loop note above
            # Redacted on purpose: this is the single most sensitive category
            # of message the system handles, and the admin dashboard's
            # "escalated cases" table displays the query text verbatim.
            # Logging the raw disclosure would put self-harm/abuse content
            # in plaintext, in a table anyone with dashboard access can read,
            # with no retention policy. Category/severity/timestamp are
            # still logged (that's what the analytics need), the message
            # itself is not.
            analytics_db.log_interaction("[crisis message — redacted from log]",
                                          crisis_category, "Critical",
                                          escalated=True)
            st.session_state.messages.append({"role": "assistant", "content": crisis_message})
            st.stop()

        # --- Stage 1.5: plain greeting / small talk (again) ------------------
        # Same reasoning as Stage 1 above -- covers the resumed original
        # question in the rare case it's itself a greeting.
        if is_chitchat(user_query):
            reply = llm_engine.generate_chitchat_reply(user_query)
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.stop()

        # --- Stage 1.7: typo tolerance --------------------------------------
        # Correct likely typos in domain words BEFORE classification and
        # retrieval, so a mistyped message still gets a real answer instead
        # of "I don't understand". The original text is what's shown/logged;
        # the normalized version only feeds the pipeline internals.
        normalized_query = normalize_query(user_query)

        # --- Stage 2: category + severity classification -------------------
        # Trained neural net first (the actual "AI" doing the classifying,
        # not keyword rules). Falls back to rules only if the model hasn't
        # been trained yet, or the LLM if even that comes back unsure.
        classifier_source = "unclassified"
        confidence = None
        p_high = None
        if neural_classifier.is_available():
            category, severity, confidence, p_high = neural_classifier.classify_full(normalized_query)
            classifier_source = f"neural_net (confidence {confidence:.0%})"
            if confidence < 0.4:
                category = "Unclassified"
        else:
            triage = rule_based_classify(normalized_query)
            category, severity = triage.category, triage.severity
            classifier_source = triage.matched_rule

        if category == "Unclassified":
            category, severity = llm_engine.classify_with_llm(normalized_query)
            classifier_source = "llm_fallback"
            confidence = None
            p_high = None

        # --- Stage 2.2: harassment safety net + escalation decision --------
        # Decided ONCE, here, before any early exit, so the flag can no longer
        # be lost by the "out of scope" / "not in knowledge base" branches.
        category, severity, escalated, escalation_reason, override_note = finalize_triage(
            normalized_query, category, severity, confidence, p_high)
        if override_note:
            classifier_source += f" + {override_note}"
        if escalated:
            classifier_source += f" | escalated: {escalation_reason}"

        # The classifier saying "Out of Scope" is not final. It only sees the
        # question's wording, so KB content it was never trained on (IT/portal
        # help, grading rules, certificates...) can be mislabelled. Search the
        # knowledge base first; hand off to an admin only if nothing matches.
        is_oos = category == "Out of Scope"

        # --- Stage 2.5: nationality gate for fee-specific questions ---------
        # Only pauses when the question is actually about a number that
        # differs by residency status, and only asks once per session.
        if (category in config.NATIONALITY_SENSITIVE_CATEGORIES
                and st.session_state.nationality is None
                and any(h in normalized_query.lower() for h in _FEE_SPECIFIC_HINTS)):
            msg = ("Quick question before I point you to the right figures — "
                   "are you a **Ghanaian** or **international** student? Fees and "
                   "some payment details differ between the two.")
            st.markdown(msg)
            st.session_state.awaiting_nationality = True
            st.session_state.pending_query = user_query
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.stop()

        if _is_advisor_question(normalized_query) and not st.session_state.awaiting_department:
            dept_list = "\n".join(f"- {d}" for d in DEPARTMENTS)
            msg = f"That depends on your department. Which of these are you in?\n\n{dept_list}"
            st.markdown(msg)
            st.session_state.awaiting_department = True
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.stop()

        with st.spinner("Searching official welfare policy documents..."):
            results = _search_with_recovery(vectorstore, normalized_query, k=3)

        _retrieval_cutoff = st.session_state.get("tuning_retrieval_cutoff", CONFIDENCE_THRESHOLD)
        if not results or results[0][1] > _retrieval_cutoff:
            msg = llm_engine.get_not_in_kb_reply(user_query)
            msg += _ADMIN_HANDOFF
            if escalated:
                _office = action_planner.get_recommended_office(category)
                if _office:
                    msg += f"\n\n**Recommended office:** {_office['office']}\n\n{_office['note']}"
            st.markdown(msg)
            analytics_db.log_interaction(_loggable(user_query, category), category, severity,
                                          escalated=escalated,
                                          classifier_source=classifier_source, needs_review=True,
                                          escalation_reason=escalation_reason)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.stop()

        if is_oos:
            classifier_source += " + kb_rescued (classifier said Out of Scope, KB match found)"
        else:
            # Category alone is shown (e.g. "Accommodation") so the student
            # can see the system understood their topic. Severity is not
            # shown -- see the note on severity_badge().
            st.markdown(f'<span class="badge badge-category">{category}</span>', unsafe_allow_html=True)

        # Decided before generating the answer so we know whether a plan is
        # even needed -- lets us make ONE combined LLM call for the answer
        # and the action plan together, instead of two sequential ones. Each
        # local (Ollama) generation can take many seconds on CPU, so this
        # roughly halves the wait on every fully-answered question.
        office_info = None if is_oos else action_planner.get_recommended_office(category)

        if office_info:
            with st.spinner("Generating answer and next steps..."):
                answer, plan = llm_engine.generate_answer_and_plan(
                    user_query, results, category=category, nationality=st.session_state.nationality)
        else:
            with st.spinner("Generating answer..."):
                answer = llm_engine.generate_grounded_answer(
                    user_query, results, category=category, nationality=st.session_state.nationality)
            plan = ""

        # possible_crisis is deliberately soft-escalated: the student sees a
        # completely normal answer, no emergency screen (that screen is
        # reserved for the exact-phrase crisis list). But "completely normal"
        # meant the student got zero acknowledgment that anything concerning
        # came through, even when a human is now quietly reviewing their
        # message. One warm, non-alarming line closes that gap without the
        # weight of the full crisis UI.
        if escalation_reason == "possible_crisis":
            answer += ("\n\n---\n*If things ever feel like too much, support is always "
                       "available — you don't have to wait for it to get worse. "
                       "You can also reach out any time through the resources in "
                       "your welfare office.*")

        st.markdown(answer)

        # --- Agent step: decide where to route + what to do next -----------
        # This is what makes it an agent rather than a Q&A chatbot: it
        # doesn't stop at answering, it recommends a specific office and a
        # concrete next-steps plan. office_info and plan were already
        # computed above so the answer and plan could come from one LLM call.
        full_reply = answer
        if office_info:
            st.markdown(f"**Recommended office:** {office_info['office']}")
            st.caption(office_info["note"])
            full_reply += f"\n\n**Recommended office:** {office_info['office']}"
            if plan:
                st.markdown("**Your action plan:**")
                st.markdown(plan)
                full_reply += f"\n\n**Action plan:**\n{plan}"

        top_source = results[0][0].metadata.get("source", "")
        interaction_id = analytics_db.log_interaction(
            _loggable(user_query, category), category, severity, escalated=escalated,
            top_source=top_source,
            recommended_office=office_info["office"] if office_info else "",
            classifier_source=classifier_source,
            escalation_reason=escalation_reason,
        )
        _render_feedback(interaction_id)
        st.session_state.messages.append({"role": "assistant", "content": full_reply,
                                           "interaction_id": interaction_id})






