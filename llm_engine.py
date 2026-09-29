"""
Wraps whichever LLM provider is configured (Gemini or Anthropic) behind two
functions the rest of the app calls:

  generate_grounded_answer(query, retrieved_chunks) -> str
  classify_with_llm(query) -> (category, severity)

If no API key is set, both functions fall back to a deterministic template
so the app still runs end-to-end for a demo â€” it just won't have natural
generated prose. This matters for grading: the app should never crash or
go blank just because a key isn't in the environment.
"""
import json
import config

import re

# Matches an internal sourcing label like "[VERIFIED]", "[SIMULATED]", or
# "[DESIGN RULE -- not sourced from UGCCD]" at the START of an answer.
# These labels exist so the team can be honest, in the report, about which
# knowledge-base content is confirmed vs. simulated for the prototype (see
# ugccd counselling policy.md's own note on this). They must NEVER reach a
# student in the actual chat -- a student asking about feeling anxious
# should not see the literal text "[SIMULATED] Assumed to be covered
# under...". Applied to every path that can show retrieved text to a
# student: the no-LLM direct fallback, and (as a safety net, in case the
# LLM quotes the tag despite being told not to) the LLM's own output too.
_KB_TAG_RE = re.compile(r"^(\s*\[[^\]]{1,60}\]\s*)+", re.IGNORECASE)


def _strip_kb_tags(text: str) -> str:
    return _KB_TAG_RE.sub("", text).strip()


def _get_tuning(key: str, default):
    """Read a session-state tuning value safely - works outside Streamlit too."""
    try:
        import streamlit as st
        return st.session_state.get(key, default)
    except Exception:
        return default


def _call_ollama(prompt: str) -> str:
    import ollama
    client = ollama.Client(host=config.OLLAMA_HOST)
    response = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": _get_tuning("tuning_temperature", 0.3),
            "num_predict": _get_tuning("tuning_max_tokens", 800),
        },
    )
    return response["message"]["content"].strip()


def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": _get_tuning("tuning_temperature", 0.3),
            "max_output_tokens": _get_tuning("tuning_max_tokens", 800),
        },
    )
    return response.text.strip()


def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=_get_tuning("tuning_max_tokens", 800),
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def _call_llm(prompt: str) -> str:
    if config.LLM_PROVIDER == "ollama":
        return _call_ollama(prompt)
    if config.LLM_PROVIDER == "gemini":
        return _call_gemini(prompt)
    if config.LLM_PROVIDER == "anthropic":
        return _call_anthropic(prompt)
    raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")


def _strip_faq_formatting(content: str) -> str:
    """Strip leading section headings and any stacked "Q: ... A:" blocks so
    only the answer text reaches the student (no-LLM fallback path)."""
    # Strip leading pure-section headings; stop when a heading IS the Q: line.
    while True:
        match = re.match(r"^\s*#{1,6}\s*(.*)\n+", content)
        if not match:
            break
        if re.match(r"Q:\s*", match.group(1), flags=re.IGNORECASE):
            break
        content = content[match.end():]
    # Looped, not one-shot: several "## Q:" headers can be stacked before one
    # answer, so keep stripping Q:/A: blocks and bare Q: lines until none remain.
    while True:
        match = re.match(r"\s*#{0,6}\s*Q:\s*.*?\n+A:\s*", content, flags=re.IGNORECASE | re.DOTALL)
        if match:
            content = content[match.end():]
            continue
        match = re.match(r"\s*#{0,6}\s*Q:\s*.*?\n+", content, flags=re.IGNORECASE)
        if match:
            content = content[match.end():]
            continue
        break
    return content.strip()


def generate_grounded_answer(query: str, retrieved_chunks: list, category: str = None,
                              nationality: str = None) -> str:
    """retrieved_chunks: list of (doc, score) tuples from the vector store.
    category/nationality are optional context used to avoid guessing on
    fee-type questions where the real answer depends on residency status."""
    top_doc, _ = retrieved_chunks[0]

    def _direct_fallback():
        # No meta-commentary, no exposed error text, no "Based on X.md" preamble,
        # no blockquote -- just the retrieved answer presented as the answer, the
        # way a student would actually want to read it. Any real error (e.g.
        # Ollama not running) is logged server-side for debugging, never shown
        # in the chat -- a student should never see an internal exception.
        content = top_doc.page_content.strip()
        return _strip_kb_tags(_strip_faq_formatting(content))

    if not config.llm_is_configured():
        return _direct_fallback()

    context_block = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc, _ in retrieved_chunks
    )

    nationality_note = ""
    if category in config.NATIONALITY_SENSITIVE_CATEGORIES:
        if nationality:
            nationality_note = (f"\nThe student has told you they are a {nationality} student. "
                                 f"If the context gives different figures/links for Ghanaian vs "
                                 f"international students, use the ones for a {nationality} student.")
        else:
            nationality_note = (
                "\nFees and some figures in the context differ for Ghanaian vs. international "
                "students. If the student's question depends on that and they haven't told you "
                "which they are, ask them directly in one short sentence instead of guessing or "
                "listing both â€” e.g. \"Are you a Ghanaian or international student? Fees differ "
                "between the two.\"")

    prompt = f"""You are the UGBS Student Welfare Assistant, an AI intake system for
University of Ghana Business School students. Answer the student's question
using ONLY the context below. If the context does not contain the answer,
say so plainly and suggest the student contact the relevant office directly
â€” never invent policy details, deadlines, or contact information.

Answer directly and plainly, the way you'd tell a friend -- do not start
with phrases like "Based on the document" or "According to the source".
Go straight to what the student needs to do or know. Do NOT add generic
filler advice that isn't actually in the context (e.g. don't say "bring
your ID" or "bring relevant documents" unless the context specifically
says so). Keep it under 120 words, warm, and practical. The context may contain internal
sourcing labels like "[VERIFIED]" or "[SIMULATED]" in square brackets --
these are for the project team only and must NEVER appear in your answer;
write as if they are not there.{nationality_note}

CONTEXT:
{context_block}

STUDENT QUESTION:
{query}

ANSWER:"""

    try:
        # Safety net: even though the prompt tells the model to answer
        # plainly, an LLM can still quote a bracketed source-tag verbatim if
        # it appears in the context it was given. Strip it here too, not
        # just in the no-LLM fallback path.
        return _strip_kb_tags(_call_llm(prompt))
    except Exception as exc:  # network/auth errors shouldn't crash the demo
        print(f"[llm_engine] generate_grounded_answer LLM call failed: {exc}")
        return _direct_fallback()


# Marks the boundary between the answer and the action plan inside a single
# combined LLM response (see generate_answer_and_plan). Deliberately
# distinctive so it can't collide with ordinary answer text.
_PLAN_DELIMITER = "===ACTION_PLAN==="


def generate_answer_and_plan(query: str, retrieved_chunks: list, category: str = None,
                              nationality: str = None) -> tuple:
    """Produces the grounded answer AND the action plan from a SINGLE LLM
    call, instead of the two separate sequential calls that
    generate_grounded_answer() + generate_action_plan() would make. Each
    local (Ollama) generation can take many seconds on CPU, so making two of
    them back-to-back for every answered question roughly doubles the
    student's wait -- this halves it, with no change to what's shown.

    Returns (answer, plan). Falls back to the same template text the two
    separate functions already use on failure or when no LLM is configured,
    so behaviour is identical to before in the no-LLM/offline demo case.
    """
    top_doc, _ = retrieved_chunks[0]

    def _fallback_answer():
        content = top_doc.page_content.strip()
        return _strip_kb_tags(_strip_faq_formatting(content))

    def _fallback_plan():
        return ("For your exact next step, contact the recommended office directly "
                "â€” they'll be able to confirm what applies to your situation.")

    if not config.llm_is_configured():
        return _fallback_answer(), _fallback_plan()

    context_block = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc, _ in retrieved_chunks
    )

    nationality_note = ""
    if category in config.NATIONALITY_SENSITIVE_CATEGORIES:
        if nationality:
            nationality_note = (f"\nThe student has told you they are a {nationality} student. "
                                 f"If the context gives different figures/links for Ghanaian vs "
                                 f"international students, use the ones for a {nationality} student.")
        else:
            nationality_note = (
                "\nFees and some figures in the context differ for Ghanaian vs. international "
                "students. If the student's question depends on that and they haven't told you "
                "which they are, ask them directly in one short sentence instead of guessing or "
                "listing both â€” e.g. \"Are you a Ghanaian or international student? Fees differ "
                "between the two.\"")

    prompt = f"""You are the UGBS Student Welfare Assistant, an AI intake system for
University of Ghana Business School students. Using ONLY the context below,
produce TWO things for this student's question: an answer, and a short
action plan of concrete next steps.

Respond in EXACTLY this format, with nothing before, between, or after the
two parts other than the delimiter line shown:

<answer text>
{_PLAN_DELIMITER}
<action plan text>

For the ANSWER part: answer directly and plainly, the way you'd tell a
friend -- do not start with phrases like "Based on the document" or
"According to the source". Go straight to what the student needs to do or
know. Do NOT add generic filler advice that isn't actually in the context
(e.g. don't say "bring your ID" or "bring relevant documents" unless the
context specifically says so). Keep it under 120 words, warm, and
practical. If the context does not contain the answer, say so plainly and
suggest the student contact the relevant office directly -- never invent
policy details, deadlines, or contact information.

For the ACTION PLAN part: write a short numbered list (2-4 steps) of
exactly what the student should do next -- concrete actions like "log into
the STS portal", "submit form X", "visit office Y" -- not vague advice. Do
NOT invent deadlines, phone numbers, or requirements that aren't in the
context. Do NOT pad the list with generic filler that isn't actually stated
unless the context specifically mentions it. If there is genuinely only one
real step, write one step -- don't stretch it to hit a minimum count.

The context may contain internal sourcing labels like "[VERIFIED]" or
"[SIMULATED]" in square brackets -- these are for the project team only and
must NEVER appear in either part of your response; write as if they are
not there.{nationality_note}

CONTEXT:
{context_block}

STUDENT QUESTION:
{query}"""

    try:
        raw = _strip_kb_tags(_call_llm(prompt))
        if _PLAN_DELIMITER in raw:
            answer_part, plan_part = raw.split(_PLAN_DELIMITER, 1)
        else:
            # Model didn't follow the delimiter format -- treat the whole
            # response as the answer rather than silently dropping it, and
            # fall back to the honest "contact the office" plan text.
            answer_part, plan_part = raw, ""
        answer_part = answer_part.strip()
        plan_part = _strip_kb_tags(plan_part.strip())
        if not answer_part:
            answer_part = _fallback_answer()
        if not plan_part:
            plan_part = _fallback_plan()
        return answer_part, plan_part
    except Exception as exc:  # network/auth errors shouldn't crash the demo
        print(f"[llm_engine] generate_answer_and_plan LLM call failed: {exc}")
        return _fallback_answer(), _fallback_plan()


def generate_action_plan(query: str, category: str, retrieved_chunks: list) -> str:
    """The 'agent' part: not just an answer, but a short ordered plan of what
    the student should actually do next, grounded in the retrieved policy
    text so steps aren't invented or padded with generic filler."""
    if not retrieved_chunks:
        return ""

    if not config.llm_is_configured():
        # No generic "bring your ID" filler here either -- if we can't call
        # an LLM to extract the real next step from the context, the honest
        # fallback is to point the student to the right office rather than
        # invent steps. The underlying source file is still available to
        # admins via analytics_db (top_source, logged separately in
        # user_interface.py) -- it's just never named in what the student
        # reads, same as the main LLM path already does.
        return ("For your exact next step, contact the recommended office directly "
                "â€” they'll be able to confirm what applies to your situation.")

    context_block = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc, _ in retrieved_chunks
    )

    prompt = f"""A UGBS student raised this welfare issue (category: {category}):
"{query}"

Using ONLY the policy context below, write a short numbered action plan
(2-4 steps) of exactly what the student should do next â€” concrete actions
like "log into the STS portal", "submit form X", "visit office Y" â€” not
vague advice. Do NOT invent deadlines, phone numbers, or requirements that
aren't in the context. Do NOT pad the list with generic filler that isn't
actually stated (e.g. "bring your ID", "bring relevant documents") unless
the context specifically mentions it. If there is genuinely only one real
step, write one step â€” don't stretch it to hit a minimum count.

CONTEXT:
{context_block}

ACTION PLAN (numbered list only):"""

    try:
        return _call_llm(prompt)
    except Exception:
        # Same as the no-LLM-configured fallback above -- no filename, no
        # "the details above" pointer, just a plain next step.
        return ("For your exact next step, contact the recommended office directly "
                "â€” they'll be able to confirm what applies to your situation.")


def get_not_in_kb_reply(query: str) -> str:
    """Used when nothing in the knowledge base is a confident match, or the
    question is genuinely out of scope. Never a dead end -- always gives a
    direct link to check, and the caller logs this for admin follow-up so
    real knowledge-base gaps get noticed and fixed over time."""
    return (
        "I don't have specific information on that in what I've been given so far. "
        f"You can check the official sites directly: [UGBS website]({config.UGBS_WEBSITE}) "
        f"or [University of Ghana website]({config.UG_WEBSITE}). "
        "I've also flagged your question so the team can add this to what I know."
    )


def generate_chitchat_reply(query: str) -> str:
    """For plain greetings/small talk (see risk_classifier.is_chitchat).
    Responds naturally and briefly, then invites the student to share
    what's going on -- instead of forcing every message through
    classification/retrieval, which produced a stiff 'out of scope'
    response to something as simple as 'hi'."""
    if not config.llm_is_configured():
        return ("Hey! I'm the UGBS Student Welfare Assistant. Tell me what's going on -- "
                "financial, academic, accommodation, mental health, careers, or BHJCR/SRC "
                "matters -- and I'll help you figure out the right next step.")

    prompt = f"""You are the UGBS Student Welfare Assistant, a friendly intake AI for
University of Ghana Business School students. The student just sent a greeting
or small talk, not a welfare question yet: "{query}"

Reply naturally and warmly in 1-2 short sentences, like a real conversation --
not a rigid template. Briefly mention you can help with things like financial
aid, academic issues, accommodation, counselling, careers, or BHJCR/SRC
matters, and invite them to share what's going on. Do not use bullet points
or headers for this reply."""

    try:
        return _call_llm(prompt)
    except Exception:
        return ("Hey! I'm the UGBS Student Welfare Assistant. Tell me what's going on -- "
                "financial, academic, accommodation, mental health, careers, or BHJCR/SRC "
                "matters -- and I'll help you figure out the right next step.")


def classify_with_llm(query: str):
    """Used only when the rule-based classifier in risk_classifier.py
    doesn't confidently match. Returns (category, severity).

    Falls back to "Out of Scope" rather than "Unclassified": "Unclassified"
    is not one of config.CATEGORIES, which meant action_planner.OFFICE_MAP
    (keyed by the real category names) had no entry for it, so a student
    silently got no office referral at all whenever this path failed. "Out
    of Scope" IS a real category with a real, honest fallback response
    (llm_engine.get_not_in_kb_reply), so failing this way degrades to
    something the rest of the app already knows how to handle."""
    if not config.llm_is_configured():
        return "Out of Scope", "Low"

    prompt = f"""Classify this student welfare message into exactly one category
from this list: {config.CATEGORIES}
and exactly one severity from this list: {config.SEVERITY_LEVELS}

Respond with ONLY valid JSON: {{"category": "...", "severity": "..."}}

MESSAGE: {query}"""

    try:
        raw = _call_llm(prompt)
        raw = raw.strip().strip("`").replace("json\n", "")
        parsed = json.loads(raw)
        category = parsed.get("category", "Out of Scope")
        severity = parsed.get("severity", "Low")
        if category not in config.CATEGORIES:
            category = "Out of Scope"
        if severity not in config.SEVERITY_LEVELS:
            severity = "Low"
        return category, severity
    except Exception as exc:
        # Was a bare "except Exception: return ..." with no logging -- silent
        # failures here were invisible next to generate_grounded_answer's
        # logged failures. Matched to that pattern for consistency.
        print(f"[llm_engine] classify_with_llm LLM call failed: {exc}")
        return "Out of Scope", "Low"
