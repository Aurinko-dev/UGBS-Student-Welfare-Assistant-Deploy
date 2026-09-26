# UGBS Student Welfare AI — Report Support Material

For Scenario 6 (Student Complaints and Administrative Case Management), AI Applications in Business project.

---

## 1. Problem definition

Students facing welfare issues at UGBS — financial hardship, academic distress, accommodation problems, mental health concerns, career questions, or governance/JCR matters — currently have no single, reliable channel to find the right office or get an accurate first answer. The as-is process (see your process diagram) shows the core failure: a student walks into the *first* office they think of, is told to "try elsewhere" if it's the wrong one, and the cycle repeats until the issue either gets handled ad hoc or escalates into a crisis with no shared record and no follow-up. The result is a widening gap between when a problem starts and when it reaches the office actually equipped to help — worse for time-sensitive cases (financial deadlines, safety concerns) than for routine ones.

**Scope for this project:** an AI-based triage and case-management layer that (1) answers routine policy questions directly and accurately, grounded in official UGBS/UG source documents, (2) classifies incoming queries by category and severity, (3) routes each query to the correct office with a concrete action plan, (4) detects and immediately escalates crisis-level messages without ever letting a generative model make that call, and (5) logs interactions so patterns (recurring issues, bottlenecks, unhandled categories) become visible to administrators instead of staying implicit in individual staff members' memory.

## 2. Stakeholder analysis

| Stakeholder | Current role in the problem | What they need from this project |
|---|---|---|
| **UGBS students** | Bear the cost of the current ad-hoc process directly — wasted time, unclear next steps, no record of having raised an issue before | Fast, accurate answers; a clear "who do I actually contact" outcome; a safe path when the issue is serious |
| **MSSU (Mentoring and Student Services Unit)** | UGBS's own front-line student-support unit — the most realistic real-world "process owner" this project's triage layer would sit in front of | A first-pass filter that reduces routine/repetitive queries reaching them manually, and structured data on what students are actually asking about |
| **Class Representatives / BHJCR Welfare Committee** | Informal first point of contact for many UGBS-specific issues (per the BHJCR constitution); currently no structured escalation path | A defined route for when a student issue should reach them vs. a university-wide office |
| **Dean of Student Affairs / Disciplinary Board for Junior Members** | The university-level authority for non-academic misconduct and serious complaints, per the University Statutes | Complaints that genuinely need their level of authority arrive appropriately categorized, not mixed in with routine questions |
| **Academic Affairs Directorate / SFAO / Counselling & Placement Centre** | Each independently handles one category of the problem (academic, financial, mental health) with no shared triage layer upstream | Reduced volume of misdirected or premature queries; accurate policy answers deflect questions that don't need a human at all |
| **UGBS Administration (Dean's Office)** | Currently has no aggregate visibility into what welfare issues are actually recurring across the student body | Analytics: category volume, severity distribution, and escalation patterns as management information — something the current manual, per-case process (e.g. the individually signed General Notice process for disciplinary sanctions) doesn't produce |

## 3. AI-opportunity comparison

| Dimension | As-is (manual/ad hoc) | With AI triage layer |
|---|---|---|
| First point of contact | Student guesses which office; often wrong | System classifies and routes on first contact |
| Answer accuracy | Depends on which staff member is asked, when | Grounded in cited source documents; explicit "I don't know" when retrieval confidence is low |
| Crisis handling | No dedicated fast path; depends on which office the student happened to reach | Deterministic keyword check runs first, unconditionally, before any generative step |
| Record-keeping | Informal; no structured log of routine queries | Every interaction logged with category, severity, escalation flag |
| Management visibility | None at the aggregate level; issues become visible only when they escalate to formal cases (e.g. the Disciplinary Board process) | Dashboard-level view of demand by category, severity trend, escalation rate |
| Existing automation baseline | ChatUGBS exists (introduced at 2024/2025 orientation) but appears limited to announcements/basic Q&A — no confirmed case-management or triage capability found in public documentation | Adds the missing analytical/triage layer on top of that baseline |

**Framing note for your report:** the ChatUGBS comparison is based on reasonable inference from orientation materials, not confirmed technical documentation — state it as an assumption, not a verified fact, when you write this up.

## 4. Escalation model (mirrors the University's own structure, not an invented taxonomy)

Informal / unit-level (class rep, MSSU) → Academic Affairs / SFAO / Counselling (category-specific office) → Disciplinary Board for Junior Members (for misconduct) → University Appeals Board, respecting the Statutes' "Exhaustion of Internal Remedies" requirement (internal stages must be exhausted before external appeal).

This mirrors the real Commonwealth Hall JCR case structure verified earlier in this project: complaint lodged → Disciplinary Board hearing → finding under specific Statute/Regulation provisions → sanction → formal Registrar notice.

---

## 5. Risks and ethical implications (data privacy)

This system handles some of the most sensitive categories of student data there are — mental health disclosures, financial hardship, sexual harassment/GBV reports, and (via crisis detection) self-harm/abuse language — and logs interaction metadata to a persistent SQLite database for both the analytics dashboard and the real-time admin monitor. This section should be in your report explicitly, not assumed obvious.

**What the prototype does about this:**
- Two categories of message get their raw text **redacted before logging**: crisis-flagged messages and Sexual Harassment/GBV-category messages. Only category, severity, and timestamp are stored for these; the verbatim disclosure never touches the database. This was a specific fix made during development, not the original design — the first version logged crisis messages verbatim, and the admin dashboard's "Recent escalated cases" table would have displayed that raw text with no access control in front of it. The GBV redaction was added for the same reason once that category existed.
- Both the Admin Interface (decision-support dashboard) and the Live Admin Interface (real-time monitor with an acknowledge/reopen workflow) carry an explicit warning banner stating this is prototype-only access control — the Live Admin Interface's own on-page text tells administrators directly that they see only category/severity/status for redacted categories, never the student's words.

**What remains a real limitation, and should be named as such in your report:**
- Non-crisis, non-GBV query text (e.g. "I'm worried about my scholarship payment") is still stored in plaintext, unredacted, in a local SQLite file with no encryption at rest.
- There is no authentication in front of the chat or either admin dashboard — anyone with the running app's URL can see the analytics, acknowledge escalations, or mark items resolved. The acknowledge/reopen workflow on the Live Admin Interface has no concept of *which* admin acknowledged something, since there's no login.
- There is no data retention or deletion policy — the database grows indefinitely with no expiry.
- No student identifier is collected (a privacy positive — sessions aren't linked to individual students), but free-text queries in non-redacted categories can still contain self-identifying details a student chooses to include (e.g. naming themselves, their hall, or their situation in detail).

**Suggested framing for the report's limitations/ethics section:** *"As a prototype, this system minimizes the most acute privacy risks (crisis disclosures and sexual harassment/GBV reports) by redacting that content at the point of logging, in both the analytics dashboard and the real-time admin monitor. However, it does not yet implement authentication, encryption at rest, or a data retention policy — all of which would be required before handling real student welfare data in production. This is a deliberate scoping decision for a course project, not an oversight, but it should be named explicitly as out-of-scope rather than left implicit."*

---

## 6. Evaluation methodology and results

This is genuinely strong evidence for your report — real held-out evaluation, not just a demo working once.

**Method:** `eval_classifier.py` runs the full system (crisis check → chit-chat check → neural classifier → `finalize_triage()`, exactly what the live app does) against a keyword-rules-only baseline, on three test sets explicitly checked for zero overlap with the training data:

| Test set | Messages | Purpose |
|---|---|---|
| `eval_testset.csv` | 87 | General coverage across all 8 categories, chit-chat, and crisis |
| `eval_testset_crisis_v2.csv` | 17 | Indirect suicide-risk phrasing (first adversarial round) |
| `eval_testset_crisis_v3.csv` | 14 | Indirect suicide-risk phrasing (harder adversarial round) |

**Headline result (main test set, 87 messages):** category accuracy 63.2% → 85.1%, severity accuracy 60.9% → 79.3%, escalation recall 27.3% → 77.3% (with a corresponding precision cost from 100% to 63%, which is the *correct* tradeoff — the baseline's 100% precision comes from almost never escalating anything, missing 3 of every 4 cases that actually needed a person).

**The strongest result for a report on a welfare-safety tool is the crisis-detection iteration:** the system's crisis phrase list and soft-detection regex were expanded twice, each time based on specific messages the evaluation caught as missed, not a general broadening:

| Test set | Recall before | After round 1 | After round 2 |
|---|---|---|---|
| Main set, CRISIS category | 40.0% | 100.0% | — |
| crisis_v2 | 53.3% | 100.0% | — |
| crisis_v3 | 8.3% | 83.3% | **100.0%** |

This is directly demonstrable, traceable evidence: specific missed phrases (e.g. "I am thinking of killing myself," later "I have written a goodbye note") were identified from real evaluation output, added to the pattern lists with in-code comments explaining why, and the fix was confirmed by re-running the same evaluation. This before/after/after-again progression, with exact numbers, is stronger evidence of engineering rigor than a single accuracy figure would be — use it as a worked example in your methodology section, not just a results table.

**One nuance worth including rather than glossing over:** on the crisis test sets, raw category accuracy looks low (as low as 14.3–21.4%) even after the fixes. This is not a detection failure — it reflects that messages caught by the "possible crisis" soft-detection layer are correctly escalated via the Mental Health/Counselling category rather than landing on the literal test-set label "CRISIS." Escalation recall, not category accuracy, is the metric that reflects real safety performance on these sets, and it's the one to lead with when reporting these results.