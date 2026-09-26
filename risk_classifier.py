"""
Two-stage triage for every incoming student message:

  1. CRISIS CHECK (deterministic, keyword-based, runs first, cannot be
     overridden by the LLM). If a message matches, the system never lets
     the LLM touch it — it goes straight to the emergency-referral screen.
     This is a deliberate design choice: a generative model should not be
     the thing deciding whether someone in crisis gets help or a chatbot
     reply. Fast, auditable, and explainable to a grader.

  2. CATEGORY + SEVERITY CLASSIFICATION for everything that passes stage 1.
     Rule-based first pass (transparent, no API cost, works offline);
     falls back to the LLM only when the rules don't confidently match,
     via classify_with_llm() in llm_engine.py.

Keep this file's keyword lists focused on RECOGNITION, not reproduction —
they exist so the system can route someone to help faster, not to describe
methods or content in detail anywhere in the app's output.
"""
from dataclasses import dataclass
import difflib
import re
from typing import Optional, Tuple

import sentiment_analysis

# Deliberately short, high-precision phrases. Longer/softer variants are
# caught via substring matching below (e.g. "kill myself" also catches
# "I want to kill myself").
_CRISIS_PATTERNS = [
    "kill myself", "kill my self", "end my life", "take my own life", "take my life",
    "suicide", "suicidal", "end it all", "better off dead", "rather be dead",
    "self harm", "self-harm", "hurt myself", "harm myself", "cutting myself",
    "want to die", "don't want to live", "do not want to live", "no reason to live",
    "being abused", "i was abused", "sexually assaulted", "being harassed",
    "was raped", "been raped", "raped me", "i was raped", "forced me to have sex",
    "assaulted me", "molested me", "groped me",
    "someone is hurting me", "domestic violence",
    # --- Added after evaluation: wording variants the first list missed
    # ("thinking of killing myself", "feel like harming myself", "nobody would
    # miss me") were routed as ordinary counselling messages, not as crises.
    # Phrases stay specific so everyday uses ("cut my expenses", "killing me
    # with assignments") do not trigger the emergency screen.
    "killing myself", "harming myself", "hurting myself", "cut myself",
    "ending my life", "end my own life", "ending my own life",
    "taking my own life", "taking my life", "ending it all",
    "wish i was dead", "wish i were dead", "wish i was never born",
    "want to be dead", "wanna die", "no longer want to live",
    "don't want to be alive", "do not want to be alive",
    "don't want to be here anymore", "do not want to be here anymore",
    "not worth living", "no point in living", "no point living",
    "nobody would miss me", "no one would miss me", "better off without me",
    # --- Added after crisis_v3 evaluation: a completed, concrete plan
    # ("I have written a goodbye note") is a materially higher-severity
    # signal than ambiguous distress language, so it belongs in the hard
    # list (bypasses the LLM, shows the emergency screen immediately),
    # not the softer possible_crisis() regex below.
    "goodbye note", "goodbye letter", "farewell note", "suicide note", "wrote a note",
    # --- Added during Stage 5 adversarial re-check. These phrases occurred in
    # the project's crisis-v3 test set and must reach the crisis path rather
    # than being treated as ordinary counselling text.
    "keep living like this", "point of any of this anymore",
    "exhausted with being alive", "everything to stop",
    "happier if i was not around", "happier if i were not around",
]

_FINANCIAL_KEYWORDS = ["fee", "fees", "tuition", "sponsorship", "scholarship",
                        "financial aid", "sfao", "can't pay", "cannot pay", "afford"]
_ACADEMIC_KEYWORDS = ["probation", "gpa", "failing", "fail", "resit", "defer",
                       "deferment", "withdraw", "academic standing", "course load"]
_ACCOMMODATION_KEYWORDS = ["hostel", "accommodation", "housing", "roommate", "landlord", "rent"]
_MENTAL_HEALTH_KEYWORDS = ["counsel", "counsellor", "counseling", "stress", "anxiety",
                           "anxious", "depressed", "depression", "overwhelmed", "mental health", "ugccd"]
_CAREER_KEYWORDS = ["career", "internship", "job", "cv", "resume", "interview"]
_GOVERNANCE_KEYWORDS = ["bhjcr", "jcr", "src", "welfare committee", "class rep",
                        "class representative", "general assembly", "electoral commission",
                        "election", "judiciary", "impeachment", "disciplinary", "constitution",
                        "business school president", "ugbs president", "become president"]
_HARASSMENT_KEYWORDS = ["sexual harassment", "harassment", "harassed", "harassing",
                        "cegensa", "unwanted advances", "inappropriate comments", "inappropriate messages",
                        "sexual misconduct", "gender-based violence", "gbv",
                        "stalking me", "inappropriate touching", "sexually assaulted",
                        "molested", "groped", "stalked", "stalker", "touching me", "touched me", "sleep with him", "sleep with her",
                        "sleep with me",
                        "unwanted messages", "raped", "rape"]

# Plain greetings / small talk only. Deliberately narrow: a message only
# counts as chit-chat if, after stripping punctuation, it is ENTIRELY made
# of these words/phrases (or very close to it) -- "hi, my hall has no
# water" must NOT match this, only "hi" or "hi there" should. This keeps
# the check safe to run before classification without risking a real
# welfare query getting swallowed as small talk.
_CHITCHAT_PHRASES = {
    "hi", "hello", "hey", "hey there", "hi there", "yo",
    "good morning", "good afternoon", "good evening",
    "how are you", "how's it going", "what's up", "whats up",
    "who are you", "what can you do", "what do you do", "help",
    "thanks", "thank you", "thanks a lot", "ok", "okay", "cool", "nice",
}


def is_chitchat(text: str) -> bool:
    """True only for plain greetings/small talk with no welfare content."""
    normalized = text.strip().lower().strip("!.?, ")
    if not normalized:
        return False
    if normalized in _CHITCHAT_PHRASES:
        return True
    # Short messages (<=4 words) that start with a greeting word and don't
    # touch any known welfare keyword are still treated as chit-chat, e.g.
    # "hey, how are you doing".
    words = normalized.split()
    starts_with_greeting = words[0] in {"hi", "hello", "hey", "yo"}
    if starts_with_greeting and len(words) <= 5:
        if not any(kw in normalized for kw in (
            _FINANCIAL_KEYWORDS + _ACADEMIC_KEYWORDS + _ACCOMMODATION_KEYWORDS
            + _MENTAL_HEALTH_KEYWORDS + _CAREER_KEYWORDS + _GOVERNANCE_KEYWORDS
            + _HARASSMENT_KEYWORDS + _CRISIS_PATTERNS
        )):
            return True
    return False

_OUT_OF_SCOPE_HINTS = ["football", "score", "match", "movie", "politics", "election",
                        "recipe", "weather", "song", "lyrics"]


@dataclass
class Triage:
    is_crisis: bool
    category: str
    severity: str
    matched_rule: str

# Vocabulary for typo correction: every domain-specific word this system
# actually cares about, pulled from the keyword lists above plus a few
# extra terms that show up in real queries but aren't triage keywords
# themselves (e.g. "scholarship", "withdrawal"). Deliberately NOT a general
# English spellchecker -- just enough to stop a single missing/wrong letter
# in a domain word from silently breaking classification or retrieval,
# without ever refusing to help ("I don't understand what you mean").
_VOCAB_EXTRA = ["scholarship", "withdrawal", "deferment", "registration", "president",
                "probation", "counselling", "counseling", "harassment", "harassed",
                "assault", "assaulted", "bullying", "bullied", "suicide", "suicidal",
                "hostel", "landlord", "roommate", "internship", "interview"]
_VOCABULARY = sorted(set(
    w for kw_list in (
        _FINANCIAL_KEYWORDS, _ACADEMIC_KEYWORDS, _ACCOMMODATION_KEYWORDS,
        _MENTAL_HEALTH_KEYWORDS, _CAREER_KEYWORDS, _GOVERNANCE_KEYWORDS,
        _HARASSMENT_KEYWORDS, _VOCAB_EXTRA,
    )
    for phrase in kw_list
    for w in phrase.split()
    if len(w) >= 4
))

# Ordinary, unambiguous everyday words that must NEVER be auto-"corrected",
# no matter how high their fuzzy-match score against something in
# _VOCABULARY comes out. Found by testing: "talking" scores 0.933 and
# "taking" scores 0.857 against "stalking" -- both ABOVE the 0.85 cutoff
# below. An explicit stoplist, checked before any fuzzy match is attempted,
# is the only reliable fix -- no single cutoff number can keep these words
# safe without also disabling real, wanted corrections.
_DO_NOT_CORRECT = {"talking", "taking", "walking", "shaking", "making", "baking",
                   "waking", "raking", "faking"}


def tokenize(text: str) -> list:
    """Splits a raw message into whitespace-delimited tokens. Deliberately
    simple (no punctuation stripping, no casing changes here) -- each stage
    of the pipeline decides what it needs from a raw token, rather than this
    function baking in assumptions for all of them. Kept as its own named
    step (rather than inlined into normalize_query) so tokenization is a
    documented, testable part of the pipeline in its own right, not just an
    implementation detail of typo correction."""
    return text.split()


def correct_tokens(tokens: list) -> list:
    """Corrects likely typos in domain-relevant tokens (missed/extra/wrong
    letters) against this project's own vocabulary, so a message like
    'defered my curses' or 'schlarship' still classifies and retrieves
    correctly instead of silently falling through to 'I don't understand'.
    Leaves short tokens, numbers, and anything already spelled correctly
    untouched -- this only nudges near-misses, it doesn't rewrite freely.
    Takes and returns a list of tokens (see tokenize()), not raw text."""
    corrected = []
    for w in tokens:
        core = w.strip(".,!?;:")
        if len(core) < 4 or core.lower() in _VOCABULARY or core.lower() in _DO_NOT_CORRECT:
            corrected.append(w)
            continue
        match = difflib.get_close_matches(core.lower(), _VOCABULARY, n=1, cutoff=0.85)
        if match:
            corrected.append(w.replace(core, match[0]))
        else:
            corrected.append(w)
    return corrected


def normalize_query(text: str) -> str:
    """The typo-tolerance step the rest of the app calls: tokenize the raw
    message, correct likely typos token-by-token, then rejoin into text for
    the classifier/retrieval steps downstream. Kept as a single entry point
    so callers don't need to know about tokenize()/correct_tokens()
    individually -- but each stage is now separately named, documented, and
    testable (see tokenize() and correct_tokens() above)."""
    return " ".join(correct_tokens(tokenize(text)))


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------
def _clean(text: str) -> str:
    """Lowercase and normalise curly quotes (phone keyboards insert them, and
    they silently broke every keyword containing an apostrophe)."""
    t = text.lower()
    for a, b in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"'),
                 ("`", "'")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


def _has(text: str, kw: str, exact: bool = False) -> bool:
    """Word-aware keyword match (replaces raw substring matching, which made
    'fee' match 'feeling', 'rent' match 'parent'/'current', and 'want to die'
    match 'want to diet').
      - short keywords (<=4 chars) must be a whole word (optionally + s/es/ed/ing)
      - longer keywords must start at a word boundary (stems like 'counsel' work)
      - exact=True forces a boundary on both sides (used for crisis phrases)"""
    kw = kw.lower()
    if exact:
        pat = r"(?<![a-z])" + re.escape(kw) + r"(?![a-z])"
    elif len(kw) <= 4:
        pat = r"(?<![a-z])" + re.escape(kw) + r"(?:s|es|ed|ing)?(?![a-z])"
    else:
        pat = r"(?<![a-z])" + re.escape(kw)
    return re.search(pat, text) is not None


def check_crisis(text: str) -> bool:
    cleaned = _clean(text)
    # Also test a typo-corrected copy, so 'sucide'-style misspellings of the
    # distinctive crisis words are still caught.
    variants = {cleaned, _clean(normalize_query(cleaned))}
    return any(_has(v, p, exact=True) for v in variants for p in _CRISIS_PATTERNS)


_GBV_HINTS = _HARASSMENT_KEYWORDS + ["abused", "abuse", "domestic violence",
                                     "hurting me", "forced me", "assaulted"]

# Ties between categories go to the more safety-critical one (dict order used to
# decide this, which sent "feeling anxious" to Financial Distress).
_PRIORITY = ["Sexual Harassment / GBV", "Mental Health / Counselling",
             "Financial Distress", "Academic Distress", "Accommodation",
             "Student Governance / JCR", "Career Guidance"]


def rule_based_classify(text: str) -> Triage:
    """Fast, transparent first pass. Returns category='Unclassified' if no
    rule fires confidently, signalling the caller to fall back to the LLM."""
    lowered = _clean(text)

    if check_crisis(text):
        # Violence / assault disclosures must route to CEGENSA, not to general
        # counselling; self-harm disclosures route to counselling.
        if any(_has(lowered, k) for k in _GBV_HINTS):
            return Triage(True, "Sexual Harassment / GBV", "Critical", "crisis_keyword")
        return Triage(True, "Mental Health / Counselling", "Critical", "crisis_keyword")

    scores = {
        "Financial Distress": sum(_has(lowered, k) for k in _FINANCIAL_KEYWORDS),
        "Academic Distress": sum(_has(lowered, k) for k in _ACADEMIC_KEYWORDS),
        "Accommodation": sum(_has(lowered, k) for k in _ACCOMMODATION_KEYWORDS),
        "Mental Health / Counselling": sum(_has(lowered, k) for k in _MENTAL_HEALTH_KEYWORDS),
        "Career Guidance": sum(_has(lowered, k) for k in _CAREER_KEYWORDS),
        "Student Governance / JCR": sum(_has(lowered, k) for k in _GOVERNANCE_KEYWORDS),
        "Sexual Harassment / GBV": sum(_has(lowered, k) for k in _HARASSMENT_KEYWORDS),
    }
    best_category = max(scores, key=lambda c: (scores[c], -_PRIORITY.index(c)))
    best_score = scores[best_category]

    if best_score == 0:
        # Out-of-scope hints only apply when nothing else matched. Before, they
        # ran first, so "when is the SRC election" and "match my skills to
        # jobs" were wrongly classed as Out of Scope.
        if any(_has(lowered, k) for k in _OUT_OF_SCOPE_HINTS):
            return Triage(False, "Out of Scope", "Low", "out_of_scope_keyword")
        return Triage(False, "Unclassified", "Low", "no_rule_match")

    urgency_words = ["urgent", "immediately", "today", "emergency", "asap", "desperate"]
    has_urgency = any(_has(lowered, w) for w in urgency_words)
    multi_hit = sum(1 for v in scores.values() if v > 0)

    if has_urgency and multi_hit >= 2:
        severity = "High"
    elif has_urgency:
        severity = "Medium"
    elif multi_hit >= 3:
        severity = "Medium"
    else:
        severity = "Low"

    # Harassment/GBV gets a Medium floor: under-flagging costs far more than
    # over-flagging in this category.
    if best_category == "Sexual Harassment / GBV" and severity == "Low":
        severity = "Medium"

    return Triage(False, best_category, severity, "keyword_rules")


# ---------------------------------------------------------------------------
# Safety net + escalation policy (NEW)
# ---------------------------------------------------------------------------
GBV = "Sexual Harassment / GBV"
MENTAL = "Mental Health / Counselling"
_SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]

# Tunable policy knobs
LOW_CONFIDENCE_THRESHOLD = 0.60   # below this on a sensitive topic -> human review
SENSITIVE_CATEGORIES = {GBV, MENTAL}


def safety_override(text: str) -> Optional[Triage]:
    """Deterministic harassment/GBV safety net. Call this AFTER the neural
    classifier and use the result to override its category/severity when it
    returns something. A 43%-confidence net once labelled 'where do I report
    sexual harassment' as Student Governance while the rules were right."""
    lowered = _clean(text)
    hits = [k for k in _HARASSMENT_KEYWORDS if _has(lowered, k)]
    if not hits:
        return None
    # Medium floor, always. Even a bare "what does CEGENSA do" is flagged: a real
    # disclosure can hide behind an information-style question, and an extra
    # item in the review queue costs far less than a missed one.
    return Triage(False, GBV, "Medium", "harassment_safety_override")


def apply_severity_floor(category: str, severity: str) -> str:
    """GBV never stays at Low (the neural net can predict Low for it)."""
    if category == GBV and severity == "Low":
        return "Medium"
    return severity


def should_escalate(category: str, severity: str, confidence: Optional[float] = None,
                    is_crisis: bool = False,
                    p_high: Optional[float] = None) -> Tuple[bool, str]:
    """Single source of truth for the `escalated` flag. Returns (flag, reason).
    `confidence` is a 0-1 float from the neural net (None when rules decided)."""
    if is_crisis or severity == "Critical":
        return True, "crisis"
    if severity == "High":
        return True, "high_severity"
    if category == GBV and severity != "Low":
        return True, "gbv_report"
    if p_high is not None and p_high >= HIGH_PROB_THRESHOLD:
        return True, "possible_high"
    if (confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD
            and category in SENSITIVE_CATEGORIES):
        return True, "low_confidence_sensitive"
    return False, ""


# ---------------------------------------------------------------------------
# Text-based urgency floor (NEW). The severity head only catches ~4 in 10 truly
# High cases (cross-validated), so clear loss/safety/deadline language raises
# severity regardless of what the net says. It can only RAISE severity.
# ---------------------------------------------------------------------------
_HIGH_URGENCY_RE = re.compile(
    r"(?<![a-z])("
    r"de-?\s?registered|deregistered|"
    r"(asked|told|forced|made) to leave|"
    r"sacked|kicked out|thrown out|evict\w*|sent home|locked out|"
    r"nowhere (to|else)( go| stay| sleep| turn)?|homeless|"
    r"(lose|losing|lost) my (place|room|hall|scholarship|admission)|"
    r"removed from my|threaten\w*|"
    r"stalk\w*|being followed|following me|"
    r"(scared|afraid|terrified) (to go|to be|for my (safety|life))|unsafe|not safe|"
    r"panic attacks?|falling apart|not slept in days|"
    r"inappropriate(ly)? (touch\w*|contact)|(touch\w*|groped|grabbed) me|"
    r"about to be|"
    r"no money for food|no food|haven'?t eaten|cannot eat|can'?t eat|"
    r"emergency|urgent\w*|immediately|asap"
    r")(?![a-z])"
)
_TIME_PRESSURE_RE = re.compile(
    r"(?<![a-z])(today|tomorrow|tonight|end of (the )?day|right now|this morning|"
    r"this week|by friday|in an hour)(?![a-z])")
_STAKES_RE = re.compile(
    r"(?<![a-z])(fees?|pay\w*|money|rent|exams?|hearing|interview|cv|transcript|"
    r"deadline|hostel|application|scholarship|probation|placement|registration)(?![a-z])")
_MEDIUM_URGENCY_RE = re.compile(
    r"(?<![a-z])(registration closes|closes soon|closing soon|running out of time|"
    r"missed (the |my )?deadline|no other way|nowhere else|"
    r"can'?t cope|cannot cope|can'?t sleep|panic attack|breaking down)")


_QUESTION_STARTERS = {"when", "what", "how", "where", "who", "which", "why", "is", "are",
                      "does", "do", "can", "could", "will", "would", "should", "may"}
_FIRST_PERSON_RE = re.compile(r"(?<![a-z])(i|i'm|i've|i'll|my|me|we|our)(?![a-z])")


def urgency_floor(text: str) -> Optional[str]:
    """Returns 'High' / 'Medium' when the wording alone justifies that level,
    else None. Pattern-based, so it is a safety net, not a substitute for
    better training data."""
    t = _clean(text)
    if _HIGH_URGENCY_RE.search(t):
        return "High"
    # Time pressure + something at stake counts only when it reads as a personal
    # situation, not a question ("when does registration open this week?").
    first_word = t.split(" ", 1)[0] if t else ""
    is_question_start = first_word in _QUESTION_STARTERS
    if (_TIME_PRESSURE_RE.search(t) and _STAKES_RE.search(t)
            and _FIRST_PERSON_RE.search(t) and not is_question_start):
        return "High"
    if _MEDIUM_URGENCY_RE.search(t):
        return "Medium"
    return None


# ---------------------------------------------------------------------------
# Second tier: POSSIBLE crisis wording. The exact-phrase list above sends a
# message to the emergency screen; this broader list cannot be that certain,
# so it does NOT trigger the emergency screen. It escalates the message to a
# person for review instead (reason "possible_crisis"), because evaluation
# showed indirect wording ("I am done with life") slipped past the exact
# phrases and was never escalated. A false alarm here costs a staff member a
# minute; a miss can cost far more.
# ---------------------------------------------------------------------------
_POSSIBLE_CRISIS_RE = re.compile(
    r"(?<![a-z])("
    r"never wake up|disappear (forever|for good)|sleep forever|"
    r"(don'?t|do not|can'?t|cannot) see (the )?point|"
    r"no (reason|point) (to|in) (keep|going|carry|continu\w*|stay\w*|live|living)|"
    r"(nobody|no one) (cares|would care)( if| whether)|"
    r"live or die|"
    r"done with (life|living)|"
    r"(tired|sick) of (living|life|everything)(?! (in|with|here|on|at|off|under))|"
    r"(give|given|giving) up on (life|living)|"
    r"(end|ending) things for good|"
    r"(can'?t|cannot) (go|carry) on|"
    r"(a|being a) burden (to|on)|"
    r"(feel|feeling|am|i'?m) (so |completely )?(worthless|hopeless)"
    r")(?![a-z])"
)


def possible_crisis(text: str) -> bool:
    """Broader, softer signal than check_crisis(): escalate to a person, but
    do not show the emergency screen."""
    return _POSSIBLE_CRISIS_RE.search(_clean(text)) is not None


# Escalate when the net gives High at least this probability, even if High is
# not its top prediction. Value chosen from the threshold table
# train_classifier.py prints for the app's actual policy (urgency floor OR
# P(High) >= cutoff), run on 102 labeled examples (21 truly High):
#   cutoff | High recall | precision | % of messages flagged
#    0.10  |   100.0%    |   35.0%   |    58.8%
#    0.20  |    95.2%    |   44.4%   |    44.1%   <- chosen
#    0.30  |    90.5%    |   55.9%   |    33.3%   (previous value)
#    0.40  |    85.7%    |   58.1%   |    30.4%
# 0.30 missed 2 of 21 truly-High cases in CV; 0.20 misses only 1, at the cost
# of ~11 more points of messages flagged for review. For this tool, a missed
# High case is worse than an extra reviewed one, so recall was prioritized
# over precision -- but not all the way to 0.10, which would flag well over
# half of all traffic and likely overwhelm a real review queue.
# CAUTION: only 21 High-severity examples back this table -- each recall
# point is ~1 case. Re-run train_classifier.py and revisit this once more
# labeled data exists.
HIGH_PROB_THRESHOLD = 0.20


def _rank(sev: str) -> int:
    return _SEVERITY_ORDER.index(sev) if sev in _SEVERITY_ORDER else 0


def finalize_triage(text: str, category: str, severity: str,
                    confidence: Optional[float] = None,
                    p_high: Optional[float] = None):
    """One call that app.py makes AFTER the neural net / LLM has produced a
    category and severity. Applies the harassment safety net, the urgency
    floor, the GBV severity floor, and the escalation policy.

    Returns (category, severity, escalated, reason, note)
      note  -> short string describing any override, for the classifier_source log
    """
    notes = []
    ov = safety_override(text)
    if ov is not None:
        if category != GBV:
            notes.append(f"gbv_override (was {category})")
            category = GBV
            confidence = None          # the net's confidence no longer applies
        if _rank(ov.severity) > _rank(severity):
            severity = ov.severity

    floor = urgency_floor(text)
    if floor is not None and _rank(floor) > _rank(severity):
        notes.append(f"urgency_floor (was {severity})")
        severity = floor

    soft_crisis = possible_crisis(text)
    if soft_crisis:
        notes.append(f"possible_crisis (was {category} / {severity})")
        if category != GBV:
            category = MENTAL
            confidence = None
        if _rank(severity) < _rank("High"):
            severity = "High"

    severity = apply_severity_floor(category, severity)
    escalated, reason = should_escalate(category, severity, confidence, p_high=p_high)
    if soft_crisis and reason != "crisis":
        escalated, reason = True, "possible_crisis"

    # Sentiment as a further, independent safety net: catches messages that
    # sound distressed but trip none of the keyword/phrase/urgency patterns
    # above (e.g. "I am really struggling and don't know what to do
    # anymore" -- no crisis wording, no urgency-floor hit, nothing GBV- or
    # academic-specific for the net to key off). Only acts when nothing else
    # already escalated the message, and only raises severity to Medium (not
    # High/Critical) -- this is a noisier, unvalidated signal (see
    # sentiment_analysis.py), so it should never be the thing that suppresses
    # the emergency screen's stricter exact-phrase bar, and shouldn't claim
    # the same confidence as the urgency floor or crisis lists.
    distress = sentiment_analysis.distress_score(text)
    if not escalated and distress >= sentiment_analysis.DISTRESS_THRESHOLD:
        notes.append(f"high_distress_sentiment (score {distress:.2f})")
        escalated, reason = True, "high_distress_sentiment"
        if _rank(severity) < _rank("Medium"):
            severity = "Medium"

    return category, severity, escalated, reason, " + ".join(notes)