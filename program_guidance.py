"""
Programme guidance: a short questionnaire that suggests options to consider.

Two modes share one scoring engine:
  "ugbs"  the 9 BSc Administration options at the University of Ghana Business
          School (aimed at Level 200 students choosing an option for Level 300).
          Uses the first 7 questions.
  "all"   every major in the Humanities Handbook (Business, Languages, Performing
          Arts, Arts, Law, Social Sciences, Education, Information Studies), for
          students exploring more widely. Uses all 17 questions.

How it works (deliberately simple and explainable):
  1. The student answers multiple-choice questions.
  2. Every answer adds points to some of the options (see QUESTIONS).
  3. Each option's score is divided by the most it could have scored, so options
     that appear in fewer answers are not unfairly penalised.
  4. The top three options are shown, each with the answers that pushed it up.

It is a starting point, not a prediction or a decision. The scoring table is a set
of reasonable judgements by the project team, based on the department and
programme names in the UGBS pages and the 2017 Humanities Handbook. It should be
reviewed by an academic advisor before real use. The fit values are a ranking aid,
not a measure of how well a student will do. This tool does not reproduce course
descriptions, entry requirements or credit structures; students should check the
Handbook or the department office.

Privacy: the only thing logged is WHICH options were suggested (no answers, no
name, no identifier).

Usage in the chat page:   program_guidance.render("ugbs")   or   render("all")
"""

OPTIONS = {
    # --- Business School: BSc Administration options ---
    "ACC": dict(name="Accounting", dept="the Department of Accounting",
                blurb="Financial reporting, auditing, taxation and accounting information systems. "
                      "A foundation for professional accounting qualifications."),
    "BNF": dict(name="Banking and Finance", dept="the Department of Finance",
                blurb="Banking, financial markets, investment and corporate finance."),
    "INS": dict(name="Insurance", dept="the Department of Finance",
                blurb="Insurance principles, risk management and related financial services."),
    "MKT": dict(name="Marketing", dept="the Department of Marketing and Entrepreneurship",
                blurb="Consumer behaviour, marketing strategy, communication, sales and retail."),
    "ECM": dict(name="E-Commerce and Customer Management",
                dept="the Department of Marketing and Entrepreneurship",
                blurb="Electronic business, digital commerce and customer management."),
    "HRM": dict(name="Human Resource Management",
                dept="the Department of Organisation and Human Resource Management",
                blurb="Managing people in organisations: HR practice, behaviour, labour law and industrial relations."),
    "HSM": dict(name="Health Services Management", dept="the Health Services Management department",
                blurb="Managing hospitals, clinics and other health organisations."),
    "PAD": dict(name="Public Administration", dept="the Public Administration department",
                blurb="Managing public organisations: governance, policy implementation and public-sector operations."),
    "ANA": dict(name="Analytics (Business Analytics)",
                dept="the Department of Operations and Management Information Systems (OMIS)",
                blurb="Using data, analytical methods and technology to support business decisions."),

    # --- School of Languages ---
    "ENG": dict(name="English", dept="the Department of English",
                blurb="Language, literature and communication: close reading, literary analysis and writing."),
    "LIN": dict(name="Linguistics", dept="the Department of Linguistics",
                blurb="The scientific study of language: sound systems, grammar, meaning and language change."),
    "GLS": dict(name="Ghanaian Language Studies", dept="the Department of Linguistics",
                blurb="In-depth study of a Ghanaian language, its literature, culture and use in society."),
    "ARA": dict(name="Arabic", dept="the Department of Modern Languages",
                blurb="Arabic language, literature and the cultures of the Arabic-speaking world."),
    "CHN": dict(name="Chinese", dept="the Department of Modern Languages",
                blurb="Chinese language, literature and the culture and society of China."),
    "KIS": dict(name="Kiswahili", dept="the Department of Modern Languages",
                blurb="Kiswahili language, literature and East African culture and society."),
    "RUS": dict(name="Russian", dept="the Department of Modern Languages",
                blurb="Russian language, literature and the culture and history of the Russian-speaking world."),
    "SPA": dict(name="Spanish", dept="the Department of Modern Languages",
                blurb="Spanish language, literature and the culture of Spanish-speaking countries."),
    "FRE": dict(name="French", dept="the Department of French",
                blurb="French language, literature and Francophone culture, with strong writing and speaking practice."),
    "TRA": dict(name="Translation", dept="the Department of French",
                blurb="Practical training in translating between languages for professional and institutional use."),

    # --- School of Performing Arts ---
    "DAN": dict(name="Dance Studies", dept="the School of Performing Arts",
                blurb="Choreography, dance technique, and the theory and history of dance."),
    "MUS": dict(name="Music", dept="the School of Performing Arts",
                blurb="Music performance, composition and theory, across African and Western traditions."),
    "THE": dict(name="Theatre Arts", dept="the School of Performing Arts",
                blurb="Acting, directing, playwriting and the theory and production of theatre."),

    # --- School of Arts ---
    "ARC": dict(name="Archaeology and Heritage Studies", dept="the Department of Archaeology and Heritage Studies",
                blurb="Studying past societies through material remains, sites, and the management of heritage."),
    "HIS": dict(name="History", dept="the Department of History",
                blurb="The study of past societies and events, and how to interpret and argue from historical evidence."),
    "PHI": dict(name="Philosophy and Classics", dept="the Department of Philosophy and Classics",
                blurb="Critical thinking about ethics, knowledge, logic and existence, alongside classical thought."),
    "REL": dict(name="Study of Religions", dept="the Department for the Study of Religions",
                blurb="The comparative study of religious traditions, beliefs and practices."),

    # --- School of Law ---
    "LAW": dict(name="Law", dept="the School of Law",
                blurb="Legal principles and reasoning, and the structure of the legal system, as a foundation for further legal study."),

    # --- School of Social Sciences ---
    "ECO": dict(name="Economics", dept="the Department of Economics",
                blurb="How individuals, firms and governments make choices, and how economies function and grow."),
    "GEO": dict(name="Geography and Resource Development", dept="the Department of Geography and Resource Development",
                blurb="The study of places and environments, and how natural and human resources are used and managed."),
    "POL": dict(name="Political Science", dept="the Department of Political Science",
                blurb="Government, political systems, public policy and international relations."),
    "PSY": dict(name="Psychology", dept="the Department of Psychology",
                blurb="The scientific study of the human mind and behaviour, including cognition, development and mental health."),
    "SOC": dict(name="Sociology", dept="the Department of Sociology",
                blurb="How societies are organised, and the study of social relationships, institutions and change."),
    "SWK": dict(name="Social Work", dept="the Department of Social Work",
                blurb="Practical training to support individuals, families and communities facing social difficulties."),

    # --- School of Continuing and Distance Education ---
    "ADH": dict(name="Adult Education and Human Resource Studies",
                dept="the Department of Adult Education and Human Resource Studies",
                blurb="Lifelong learning, community education and human resource development."),

    # --- School of Information and Communication Studies ---
    "INF": dict(name="Information Studies", dept="the Department of Information Studies",
                blurb="Managing and organising information and records, including library and archival practice."),

    # --- School of Education and Leadership ---
    "TED": dict(name="Teacher Education", dept="the Department of Teacher Education",
                blurb="Preparation for a teaching career: pedagogy, curriculum design and classroom practice."),
    "PES": dict(name="Physical Education and Sport Studies", dept="the Department of Physical Education and Sport Studies",
                blurb="The study and teaching of physical education, sport science and coaching."),
}

# Each question: id, text, then a list of (answer label, {option code: points}).
QUESTIONS = [
    # --- Business-focused questions (differentiate the 9 BSc Administration options) ---
    {"id": "work", "text": "1. Which kind of work would you enjoy most?", "options": [
        ("Analysing data to find patterns and answer business questions", {"ANA": 3, "BNF": 1, "ACC": 1, "MKT": 1}),
        ("Checking, recording and reporting financial information accurately", {"ACC": 3, "BNF": 1, "INS": 1}),
        ("Working with money, markets, banks or risk", {"BNF": 3, "INS": 2, "ACC": 1}),
        ("Understanding, developing and managing people", {"HRM": 3, "HSM": 1, "PAD": 1}),
        ("Creating campaigns, selling ideas and understanding customers", {"MKT": 3, "ECM": 2}),
        ("Building or running online businesses and digital customer services", {"ECM": 3, "MKT": 1, "ANA": 1}),
        ("Running public institutions and putting policy into practice", {"PAD": 3, "HRM": 1}),
        ("Managing hospitals, clinics or health programmes", {"HSM": 3, "PAD": 1}),
    ]},
    {"id": "maths", "text": "2. How comfortable are you with mathematics and statistics?", "options": [
        ("Very comfortable, I enjoy it", {"ANA": 3, "BNF": 2, "INS": 2, "ACC": 1, "ECO": 2}),
        ("Comfortable when I need it", {"ACC": 2, "BNF": 1, "INS": 1, "ECM": 1, "MKT": 1, "GEO": 1, "PSY": 1}),
        ("I would rather keep it to a minimum", {"HRM": 2, "PAD": 2, "HSM": 2, "MKT": 1, "ENG": 1, "HIS": 1}),
    ]},
    {"id": "tools", "text": "3. How do you feel about software and data tools (Excel, SQL, Python, dashboards)?", "options": [
        ("I enjoy them and want to go deeper", {"ANA": 3, "ECM": 2, "ACC": 1, "BNF": 1, "GEO": 1}),
        ("Happy to use them as a tool", {"ACC": 1, "BNF": 1, "MKT": 1, "ECM": 1, "INS": 1, "INF": 1}),
        ("I would rather not rely on them", {"HRM": 1, "PAD": 1, "HSM": 1, "ENG": 1, "PHI": 1}),
    ]},
    {"id": "strength", "text": "4. Which strength describes you best?", "options": [
        ("Analytical and logical", {"ANA": 2, "BNF": 2, "INS": 2, "ACC": 1, "PHI": 1, "ECO": 1}),
        ("Careful and accurate with detail", {"ACC": 3, "INS": 1, "BNF": 1, "INF": 1}),
        ("Persuasive and creative", {"MKT": 3, "ECM": 2, "ENG": 1, "THE": 1}),
        ("A good listener who understands people", {"HRM": 3, "HSM": 1, "PSY": 1, "SWK": 1}),
        ("A natural organiser and leader", {"PAD": 2, "HRM": 1, "HSM": 1, "MKT": 1, "POL": 1}),
        ("Caring and service-minded", {"HSM": 3, "PAD": 1, "HRM": 1, "SWK": 2}),
    ]},
    {"id": "where", "text": "5. Where would you most like to work after graduating?", "options": [
        ("Banks, investment firms or insurance companies", {"BNF": 3, "INS": 3, "ACC": 1}),
        ("Accounting or audit firms, or a company's finance team", {"ACC": 3, "BNF": 1}),
        ("Technology, consulting or data teams", {"ANA": 3, "ECM": 1}),
        ("Marketing agencies, retail, media or e-commerce", {"MKT": 3, "ECM": 3}),
        ("The HR or people team of any organisation", {"HRM": 3}),
        ("Government agencies, diplomacy or NGOs", {"PAD": 3, "HRM": 1, "POL": 2}),
        ("Hospitals and health organisations", {"HSM": 3}),
        ("Courts, chambers or the legal department of an organisation", {"LAW": 3}),
        ("Schools, universities or training institutions", {"TED": 3, "ADH": 1}),
        ("Museums, archives, libraries or cultural institutions", {"ARC": 2, "INF": 2, "HIS": 1}),
    ]},
    {"id": "matters", "text": "6. What matters most to you in a career?", "options": [
        ("A recognised professional qualification path", {"ACC": 3, "INS": 1, "BNF": 1, "LAW": 1}),
        ("Strong earning potential in fast-moving fields", {"BNF": 2, "ANA": 2, "INS": 1, "ECM": 1, "ECO": 1}),
        ("Creativity and variety", {"MKT": 2, "ECM": 2, "ENG": 1, "MUS": 1, "THE": 1, "DAN": 1}),
        ("Stability and public service", {"PAD": 2, "HSM": 2, "HRM": 1, "TED": 1, "POL": 1}),
        ("Making a difference to people's lives", {"HSM": 2, "HRM": 2, "PAD": 1, "SWK": 2, "PSY": 1}),
        ("Innovation and new technology", {"ANA": 3, "ECM": 2}),
    ]},
    {"id": "day", "text": "7. Which kind of day-to-day work sounds best?", "options": [
        ("Building reports, models and dashboards", {"ANA": 2, "ACC": 2, "BNF": 1, "ECO": 1}),
        ("Meeting clients and building relationships", {"MKT": 2, "INS": 1, "BNF": 1, "ECM": 1}),
        ("Leading teams and solving people problems", {"HRM": 3, "PAD": 1}),
        ("Planning and coordinating programmes or services", {"PAD": 2, "HSM": 2, "ADH": 1}),
        ("Testing ideas and launching new products or ventures", {"ECM": 2, "MKT": 2, "ANA": 1}),
    ]},

    # --- Language questions ---
    {"id": "lang_interest", "text": "8. How do you feel about literature, writing and how the English language works?",
     "options": [
        ("I love reading and analysing texts, and writing well", {"ENG": 3, "PHI": 1}),
        ("I'm more interested in how language is structured scientifically", {"LIN": 3, "GLS": 1}),
        ("I'd rather learn a completely different world language", {"ARA": 1, "CHN": 1, "KIS": 1, "RUS": 1, "SPA": 1, "FRE": 1}),
        ("Not really my main interest", {}),
    ]},
    {"id": "lang_choice", "text": "9. If you had to become fluent in a new language, which appeals most?",
     "options": [
        ("Arabic", {"ARA": 3}),
        ("Chinese", {"CHN": 3}),
        ("Kiswahili", {"KIS": 3}),
        ("Russian", {"RUS": 3}),
        ("Spanish", {"SPA": 3}),
        ("French, and I enjoy converting meaning between languages", {"FRE": 3, "TRA": 2}),
        ("A Ghanaian language (e.g. Akan, Ewe, Ga)", {"GLS": 3, "LIN": 1}),
        ("None of these particularly interest me", {}),
    ]},

    # --- Performing arts ---
    {"id": "performing", "text": "10. How do you feel about performing in front of others?", "options": [
        ("I love dancing and choreography", {"DAN": 3}),
        ("I love making or performing music", {"MUS": 3}),
        ("I love acting and theatre", {"THE": 3}),
        ("I enjoy watching performances but not performing myself", {"DAN": 1, "MUS": 1, "THE": 1}),
        ("Not really for me", {}),
    ]},

    # --- Arts / humanities ---
    {"id": "arts_pursuit", "text": "11. Which of these pursuits interests you most?", "options": [
        ("Uncovering the past through artefacts and heritage sites", {"ARC": 3}),
        ("Understanding historical events and how societies changed over time", {"HIS": 3}),
        ("Big questions about ethics, knowledge, logic and existence", {"PHI": 3}),
        ("Religion, belief systems and spirituality", {"REL": 3}),
        ("None of these especially", {}),
    ]},

    # --- Law ---
    {"id": "law_interest", "text": "12. How do you feel about law, justice and formal argument?", "options": [
        ("I'd enjoy studying and arguing about law in depth", {"LAW": 3, "PAD": 1}),
        ("Interested, but not necessarily as a full degree", {"LAW": 1, "POL": 1}),
        ("Not for me", {}),
    ]},

    # --- Social sciences ---
    {"id": "social_topic", "text": "13. Which social science topic interests you most?", "options": [
        ("How economies and markets work", {"ECO": 3, "BNF": 1, "ANA": 1}),
        ("How places, environments and resources interact", {"GEO": 3}),
        ("Politics, government and international relations", {"POL": 3, "PAD": 1}),
        ("The human mind and behaviour", {"PSY": 3}),
        ("Society, culture and social structures", {"SOC": 3}),
        ("Helping vulnerable individuals and communities directly", {"SWK": 3, "HSM": 1}),
    ]},

    # --- Education / adult education / sport ---
    {"id": "education", "text": "14. How do you feel about teaching, coaching or community education?", "options": [
        ("I'd like to become a school teacher", {"TED": 3}),
        ("I'm interested in sport, fitness and coaching", {"PES": 3}),
        ("I'm interested in adult education and community development", {"ADH": 3}),
        ("Not really my interest", {}),
    ]},

    # --- Information studies ---
    {"id": "information", "text": "15. How do you feel about organising records, libraries and information?", "options": [
        ("I enjoy organising information, records and archives", {"INF": 3}),
        ("Somewhat interested", {"INF": 1}),
        ("Not really", {}),
    ]},

    # --- Broader values / day-to-day, across humanities ---
    {"id": "broad_matters", "text": "16. Beyond business, what would matter most to you in a career?", "options": [
        ("Working with data, numbers or the environment", {"ANA": 1, "ECO": 1, "GEO": 1}),
        ("Creative expression through writing, art or performance", {"ENG": 2, "DAN": 1, "MUS": 1, "THE": 1}),
        ("Advocacy, justice or public service", {"LAW": 2, "PAD": 1, "SWK": 1, "POL": 1}),
        ("Understanding culture, history or belief systems", {"HIS": 1, "PHI": 1, "REL": 1, "ARC": 1}),
        ("Teaching and shaping other people's growth", {"TED": 2, "ADH": 1}),
        ("None of these particularly", {}),
    ]},
    {"id": "broad_day", "text": "17. Which day-to-day activity appeals most to you?", "options": [
        ("Reading and closely analysing texts", {"ENG": 2, "PHI": 1, "LIN": 1}),
        ("Fieldwork, excavation or mapping", {"ARC": 2, "GEO": 2}),
        ("Rehearsing and performing", {"DAN": 1, "MUS": 1, "THE": 1}),
        ("Debating and building arguments", {"LAW": 2, "POL": 1, "PHI": 1}),
        ("Counselling or directly supporting people", {"PSY": 2, "SWK": 2}),
        ("Organising community or training programmes", {"ADH": 2, "SOC": 1}),
    ]},
]

# Options the assistant's knowledge base covers in detail (the UGBS BSc Administration options).
UGBS_CODES = {"ACC", "BNF", "INS", "MKT", "ECM", "HRM", "HSM", "PAD", "ANA"}

# In "ugbs" mode a few answer labels are worded for the whole Handbook; tidy them.
_UGBS_LABEL_FIXES = {"Government agencies, diplomacy or NGOs": "Government agencies or NGOs"}


def _ugbs_questions() -> list:
    """The first seven questions, keeping only points for the UGBS options and
    dropping answers that lead only to non-UGBS options."""
    out = []
    for q in QUESTIONS[:7]:
        opts = []
        for label, pts in q["options"]:
            kept = {c: p for c, p in pts.items() if c in UGBS_CODES}
            if kept:
                opts.append((_UGBS_LABEL_FIXES.get(label, label), kept))
        out.append({**q, "options": opts})
    return out


MODES = {
    "ugbs": dict(
        title="🧭 Which UGBS option might suit you?",
        caption="Seven quick questions about your interests and strengths. It suggests options to "
                "consider, not a decision. Only the names of the suggested options are recorded, "
                "never your answers.",
        questions=_ugbs_questions(), codes=UGBS_CODES, strong=0.65, good=0.45),
    "all": dict(
        title="🧭 Which major might suit you?",
        caption="A longer set of questions covering every undergraduate major in the Humanities "
                "Handbook: Business, Languages, Performing Arts, Arts, Law, Social Sciences, "
                "Education and Information Studies. It suggests options to consider, not a "
                "decision. Only the names of the suggested options are recorded, never your answers.",
        questions=QUESTIONS, codes=set(OPTIONS), strong=0.55, good=0.35),
}


def _max_points(mode: str) -> dict:
    m = MODES[mode]
    return {code: sum(max(o[1].get(code, 0) for o in q["options"]) for q in m["questions"])
            for code in m["codes"]}


def recommend(answers: dict, mode: str = "ugbs", top_n: int = 3) -> list:
    """answers: {question id: index of the chosen option}. Returns the top options,
    best first, each as a dict with name, dept, blurb, fit (0-1), label and why."""
    m = MODES[mode]
    maxima = _max_points(mode)
    scored = []
    for code in m["codes"]:
        info = OPTIONS[code]
        raw, contributions = 0, []
        for q in m["questions"]:
            idx = answers.get(q["id"])
            if idx is None:
                continue
            label, pts = q["options"][idx]
            p = pts.get(code, 0)
            raw += p
            if p:
                contributions.append((p, label))
        contributions.sort(key=lambda x: -x[0])
        fit = raw / maxima[code] if maxima[code] else 0.0
        scored.append(dict(code=code, raw=raw, fit=fit, why=[l for _, l in contributions[:2]], **info))
    scored.sort(key=lambda r: (-r["fit"], -r["raw"], r["name"]))
    top = scored[:top_n]
    for r in top:
        r["label"] = ("Strong match" if r["fit"] >= m["strong"]
                      else "Good match" if r["fit"] >= m["good"] else "Worth exploring")
    if len(top) > 1 and top[0]["fit"] - top[1]["fit"] < 0.05:
        top[0]["close"] = top[1]["close"] = True
    return top


def _log(result: list, mode: str) -> None:
    """Log only which options were suggested. No answers, no identifiers."""
    try:
        import analytics_db
        names = ", ".join(r["name"] for r in result)
        analytics_db.log_interaction(
            f"[Programme guide] Suggested options: {names}", "Career Guidance", "Low",
            escalated=False, recommended_office=result[0]["dept"].replace("the ", "", 1).capitalize(),
            classifier_source=f"program_guidance_{mode}")
    except Exception:
        pass   # logging must never break the student's experience


def render(mode: str = "ugbs") -> None:
    """The questionnaire and results, for the student chat page."""
    import streamlit as st

    m = MODES[mode]
    result_key = f"guide_{mode}_result"
    st.markdown(f"#### {m['title']}")
    st.caption(m["caption"])

    result = st.session_state.get(result_key)
    if result is None:
        with st.form(f"guide_{mode}_form"):
            picks = {q["id"]: st.radio(q["text"], [o[0] for o in q["options"]], index=None,
                                       key=f"guide_{mode}_{q['id']}") for q in m["questions"]}
            submitted = st.form_submit_button("Show my options")
        if submitted:
            if any(v is None for v in picks.values()):
                st.warning("Please answer every question so the suggestions are meaningful.")
            else:
                answers = {q["id"]: [o[0] for o in q["options"]].index(picks[q["id"]])
                           for q in m["questions"]}
                result = recommend(answers, mode)
                st.session_state[result_key] = result
                _log(result, mode)
                st.rerun()
        return

    st.success("Here are the options that fit your answers best.")
    for i, r in enumerate(result, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {r['name']}** · {r['label']}")
            st.write(r["blurb"])
            if r["why"]:
                st.caption("Why it came up: " + "; ".join(f"you chose “{w}”" for w in r["why"]))
            st.caption(f"Run by {r['dept']}.")
    if result[0].get("close"):
        st.info("Your top two are very close, so it is worth reading about both.")
    if result[0]["code"] in UGBS_CODES:
        st.markdown(
            "**Next steps:** ask the department office who your course advisor is, and check the "
            "current course list and any entry requirements with the Academic Office. You can also "
            f"ask me, for example, *“What does the {result[0]['name']} option involve?”*")
    else:
        st.markdown(
            "**Next steps:** the Humanities Handbook and the department office have the course "
            "list, entry requirements and how to apply. I can answer detailed questions only about "
            "the UGBS options, so ask the department directly for this one.")
    st.caption("This is a starting point based on your own answers. It cannot predict your results "
               "or guarantee a place in an option, and it does not replace the Handbook or an "
               "academic advisor.")
    if st.button("Start again", key=f"guide_{mode}_restart"):
        for q in m["questions"]:
            st.session_state.pop(f"guide_{mode}_{q['id']}", None)
        st.session_state.pop(result_key, None)
        st.rerun()