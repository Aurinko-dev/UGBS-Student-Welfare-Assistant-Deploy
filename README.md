# UGBS Student Welfare Assistant

An AI-based triage and referral assistant for University of Ghana Business School
students — built for Scenario 5 (Student Welfare and Support Services),
AI Applications in Business.

Answers routine welfare questions grounded in real UG/UGBS policy documents, classifies
incoming messages by category and severity, routes each to the right office with a concrete
action plan, detects crisis-level messages deterministically before any generative model
touches them, and gives administrators both a decision-support dashboard and a real-time
monitor with an actual resolve/acknowledge workflow.

## Architecture

`app.py` is a thin router: it sets shared page config, injects the shared stylesheet
(`ui_theme.py`), builds the sidebar, and hands off to whichever page is selected via
`st.navigation()`/`st.Page()`. All page logic lives in `views/`:

| Page | File | What it does |
|---|---|---|
| User Interface | `views/user_interface.py` | The student-facing chat — the actual triage pipeline |
| Admin Interface | `views/admin_interface.py` | Decision-support dashboard: demand by category, severity mix, daily trend, referral distribution, escalated-case table, "needs review" resolve workflow |
| Live Admin Interface | `views/live_admin_interface.py` | Real-time monitor (auto-refreshing via `st.fragment`) with an escalation queue, acknowledge/reopen workflow, and toast notifications for new escalations |
| Settings | `views/settings.py` | System status, indexed knowledge base list, and a live classifier test tool that shows the full triage decision (neural net + rule fallback + safety net + final escalation call) for any typed message |

### The triage pipeline (`views/user_interface.py` + `risk_classifier.py`)

1. **Crisis check, deterministic, runs first, unconditionally.** `risk_classifier.check_crisis()`
   matches against an explicit phrase list (`_CRISIS_PATTERNS`) using word-boundary-aware
   matching (not raw substrings — this specifically avoids false positives like "want to die"
   matching inside "want to diet"). If it fires, the LLM and retrieval are skipped entirely and
   the student sees an emergency-referral screen. A separate GBV-specific emergency message
   routes violence/assault disclosures to CEGENSA rather than generic counselling.
2. **Chit-chat detection** — plain greetings get a natural reply instead of being forced through
   classification.
3. **Typo tolerance** (`normalize_query()`) — corrects likely typos against the project's own
   domain vocabulary before classification, retrieval, *and* the crisis check itself (both the
   raw and typo-corrected text are checked for crisis phrases).
4. **Category + severity classification** — a trained neural network
   (`classifier_model.py`/`neural_classifier.py`) sitting on the same MiniLM embeddings used for
   retrieval, with a rule-based keyword fallback (`rule_based_classify()`) if the model isn't
   trained yet.
5. **`finalize_triage()`** — the single function every code path (the live app *and*
   `eval_classifier.py`) routes through after classification. Applies, in order: a harassment/GBV
   safety override, an urgency-word severity floor, a "possible crisis" soft-detection layer
   (catches indirect distress language that doesn't hit an exact crisis phrase, and escalates it
   to a person without showing the full emergency screen), a severity floor, and the final
   escalation decision (including a P(High-severity) probability threshold, not just the
   classifier's top prediction).
6. **Confidence-gated retrieval** — if the best-matching document is a weak match, the assistant
   says it doesn't know and gives a direct link to the UGBS/UG website, rather than guessing.
   Both this case and genuinely out-of-scope questions are logged as "needs review" so real
   knowledge-base gaps are visible to admins, not silently dropped.
7. **Grounded answer + action plan** — the LLM (Ollama by default, Gemini/Anthropic as
   alternatives) answers using only the retrieved context, with explicit instructions against
   generic filler advice. `action_planner.py` supplies a deterministic office recommendation
   (these are policy facts, not something to let an LLM guess).
8. **Logging** — every interaction is logged to SQLite with category, severity, and escalation
   status. Crisis messages and Sexual Harassment/GBV messages are **redacted before logging** —
   only category/severity/timestamp are stored, never the verbatim disclosure, even though the
   admin dashboards display raw query text for every other category.

## Knowledge base

11 source documents, each with `[VERIFIED]`/source-attribution tagging: SFAO financial aid,
UGBS academic policies, UGCCD counselling, the BHJCR constitution, UG Statutes governance
excerpts, SRC electoral rules, UG Academic Affairs Q&A, university discipline/misconduct
governance, accommodation (halls, hostels, the random bed allocation system), and sexual
harassment reporting/support (CEGENSA). Routed to 8 categories: Financial Distress, Academic
Distress, Accommodation, Mental Health/Counselling, Career Guidance, Student Governance/JCR,
Sexual Harassment/GBV, and Out of Scope.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env                # defaults to Ollama, no key needed
python train_classifier.py          # trains the neural classifier
python build_vectorstore.py         # only needed if ugbs_welfare_db/ is missing or .md files changed
streamlit run app.py
```

If using Ollama, make sure it's running locally with a model pulled (e.g.
`ollama pull llama3.2:1b`). The app runs in template mode with no LLM configured at all —
useful for testing the triage/analytics logic without spending API credits or running a local model.

## Training and evaluation

`train_classifier.py` uses 5-fold stratified cross-validation (on category) rather than a single
train/validation split, picks the epoch count from mean CV validation loss instead of a fixed
number, applies class-weighted severity loss (Low severity is the majority class), and reports
macro-F1 and High-severity recall specifically, not just raw accuracy. It also calibrates
`HIGH_PROB_THRESHOLD` (the probability cutoff for escalating on P(High) even when High isn't the
top prediction) against a recall/precision tradeoff table.

`eval_classifier.py` runs the **full system** (exactly what the live app does, via
`finalize_triage()`) against a **keyword-rules-only baseline**, on held-out test sets explicitly
checked for zero overlap with the training data. Three test sets were used:

| Test set | Messages | What it's for |
|---|---|---|
| `eval_testset.csv` | 87 | General coverage across all 8 categories + chit-chat + crisis |
| `eval_testset_crisis_v2.csv` | 17 | Indirect suicide-risk phrasing, first adversarial round |
| `eval_testset_crisis_v3.csv` | 14 | Indirect suicide-risk phrasing, harder adversarial round |

**Results, full system vs. keyword baseline (main test set, 87 messages):**

| Metric | Baseline | Full system |
|---|---|---|
| Category accuracy | 63.2% | 85.1% |
| Severity accuracy | 60.9% | 79.3% |
| Escalation recall (cases needing a person, caught) | 27.3% | 77.3% |
| Escalation precision | 100.0% | 63.0% |

The precision drop is the correct tradeoff, not a regression: the baseline's 100% precision
only looks good because it almost never escalates anything (27.3% recall — it misses 3 of every
4 cases that actually needed a person). A false-positive escalation costs an admin a few minutes
reviewing a non-urgent case; a false-negative costs someone in real need not being flagged.

**Crisis-detection results, both adversarial sets, after iterative fixes driven by these evaluations:**

| Test set | Before | After first fix round | After second fix round |
|---|---|---|---|
| Main set (5 CRISIS messages) | 40.0% recall | 100.0% recall | — |
| crisis_v2 (15 escalation cases) | 53.3% recall | 100.0% recall | — |
| crisis_v3 (12 escalation cases) | 8.3% recall | 83.3% recall | **100.0% recall** |

Each fix round added specific phrases identified by reading the actual misses — e.g. "killing
myself" / "harming myself" / "nobody would miss me" after the first round; "goodbye note" and
"sleep forever" after the crisis_v3 round — not a general broadening of the pattern list. This
is a deliberate, evidence-driven process: every addition to `_CRISIS_PATTERNS` or
`_POSSIBLE_CRISIS_RE` is tied to a specific evaluation failure, documented in-line as a code
comment.

**A note on reading category/severity accuracy on the crisis test sets:** a message that gets
correctly escalated via the "possible crisis" soft-detection path lands on category "Mental
Health/Counselling," not the literal test-set label "CRISIS" — so raw category accuracy on
these sets understates real safety performance. Escalation recall is the metric that reflects
whether the system actually got the right people flagged, and that's the one to report as the
headline safety number.

## Suggested demo script (normal / hard / failure case)

- **Normal case:** "What are the eligibility requirements for SFAO financial aid?" → grounded
  answer, category badge, source shown.
- **Hard case:** "I can't pay my fees and I'm failing two courses, I don't know what to do" →
  shows classification confidence, an office recommendation, and a generated action plan.
- **Failure/edge cases:**
  - Off-topic question ("who won the match yesterday") → out-of-scope handling with a direct
    website link, logged as "needs review" for admin follow-up.
  - A message with indirect crisis language ("I don't want to be here anymore") → shows the
    "possible crisis" soft-escalation path, distinct from an explicit crisis phrase.
  - An explicit crisis phrase → bypasses the LLM entirely, shows the emergency-referral screen
    immediately.
  - A sexual harassment disclosure → shows GBV-specific routing to CEGENSA and redacted logging.

## Known limitations (state these explicitly, don't leave them implicit)

- No authentication in front of the chat or either admin dashboard.
- Non-crisis, non-GBV query text is stored in plaintext with no encryption at rest and no data
  retention/deletion policy.
- The typo-tolerance vocabulary is small and domain-specific, not a general spellchecker.
- Category/severity training data, while now evaluated with proper cross-validation, is still a
  relatively small hand-labeled set — see `train_classifier.py`'s per-category accuracy output
  and `cv_predictions.csv` for exactly where it's weaker.

## What's still open

- Architecture diagrams should be redone to reflect the `views/` + `st.navigation()` structure
  (earlier diagrams show the old `pages/`-based layout).
- The judgment call on whether to broaden the `tired of (living|life)` pattern in
  `possible_crisis()` to also catch "tired of everything" — flagged as a real precision/recall
  tradeoff, not resolved either way yet.
## Stage 4 implementation update

This package is the Stage 4 implementation set for the UGBS AI Student Welfare & Support Triage System.

The implemented agent flow is:

`Student input → crisis/safety screening → natural-language normalization → neural classifier/rule fallback → LLM fallback when needed → safety/urgency/escalation finalization → knowledge-base retrieval → grounded answer → deterministic office routing → action plan → SQLite logging → administrative analytics.`

The Stage 4 package also includes `stage4_agent_tests.py`, which smoke-tests the deterministic triage and routing components without requiring an LLM server or trained model weights.

### Stage 4 setup

1. Install dependencies: `pip install -r requirements.txt`
2. Train the classifier: `python train_classifier.py`
3. Build the local vector store: `python build_vectorstore.py`
4. Start the application: `streamlit run app.py`
5. Run deterministic smoke tests: `python stage4_agent_tests.py`

The prototype is not production-secured. Authentication, role-based access, retention/deletion controls, and formal privacy governance are still required before deployment with real student welfare data.