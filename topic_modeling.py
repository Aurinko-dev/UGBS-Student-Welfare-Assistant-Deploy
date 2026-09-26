"""
Topic modeling: looks across the admin's "needs review" queue (questions the
system couldn't confidently answer from the knowledge base -- see
analytics_db.get_unresolved) and finds recurring themes automatically,
without being told what to look for. This is deliberately unsupervised and
independent of the project's 8 fixed categories: the whole point is to
surface patterns the fixed category list doesn't capture (e.g. a cluster of
questions about a specific new policy, or a particular office, that don't
map cleanly onto "Academic Distress" vs "Financial Distress" etc.).

Approach: TF-IDF over the flagged question text, then NMF (Non-negative
Matrix Factorization) to find topics. NMF was chosen over LDA for this
project because it behaves better on the small, short-text corpora this
tool will realistically have (tens of flagged questions, not thousands) and
its topics are easier to read off directly as ranked keyword lists -- which
is what an admin skimming a dashboard panel actually needs. Both come from
scikit-learn, already a project dependency; no new model download.

This is exploratory admin tooling, not something shown to students, and not
a replacement for the classifier's fixed categories -- it's meant to sit
alongside them and catch what they miss.
"""
from typing import List, Dict, Optional

# sklearn is imported lazily, inside discover_topics(), rather than at
# module load time. On some locked-down machines (university-managed
# Windows installs with an Application Control / WDAC policy, for example)
# sklearn's compiled extensions can be blocked from loading -- if that
# import happened here at the top of the file, it would take down the
# entire admin page on startup rather than just this one feature.

# Below this many flagged questions, topic modeling doesn't produce anything
# meaningful -- too few documents for TF-IDF/NMF to find real co-occurrence
# patterns rather than noise. The admin panel should show a plain "not
# enough data yet" message below this, rather than a table of topics built
# from 3 unrelated questions.
MIN_DOCUMENTS = 5


def discover_topics(texts: List[str], n_topics: Optional[int] = None,
                    n_top_words: int = 6) -> Dict:
    """Runs TF-IDF + NMF over `texts` (one string per flagged/unresolved
    question) and returns discovered topics with their top keywords and
    which input documents belong most strongly to each.

    Returns a dict:
      {"ok": False, "reason": "..."}                          -- too little data
      {"ok": True, "topics": [
          {"id": 0, "keywords": [...], "doc_indices": [...], "weight": ...},
          ...
      ]}
    `doc_indices` are positions into the `texts` list the caller passed in,
    so the caller can map back to full rows (timestamp, category, etc.) if
    it wants to show example questions per topic.
    """
    texts = [t for t in texts if t and t.strip()]
    if len(texts) < MIN_DOCUMENTS:
        return {"ok": False,
                "reason": f"Only {len(texts)} flagged question(s) logged so far -- "
                          f"need at least {MIN_DOCUMENTS} before topic patterns are "
                          f"meaningful rather than noise."}

    try:
        from sklearn.decomposition import NMF
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as e:
        return {"ok": False,
                "reason": "Topic modeling needs scikit-learn, which failed to load on this "
                          "machine (" + str(e) + "). On a university-managed Windows install "
                          "this is often an Application Control / WDAC policy blocking "
                          "scikit-learn's compiled files -- ask IT to allowlist the sklearn "
                          "folder inside this project's venv."}

    # Cap topic count sensibly for small corpora: no point asking for more
    # topics than roughly a third of the documents, and never more than 6
    # (an admin skimming a dashboard can't usefully parse more clusters
    # than that anyway).
    if n_topics is None:
        n_topics = max(2, min(6, len(texts) // 2))
    n_topics = min(n_topics, len(texts) - 1)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.95,   # drop terms in almost every doc -- not distinguishing
        min_df=1,      # keep rare terms -- corpus is small, can't afford min_df=2
        ngram_range=(1, 2),  # unigrams + bigrams: "financial aid" is more
                              # readable as a topic keyword than "financial"
                              # and "aid" separately
    )
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        # Can happen if every document is pure stopwords/too short after
        # cleaning -- vectorizer ends up with an empty vocabulary.
        return {"ok": False,
                "reason": "Flagged questions didn't have enough distinct wording "
                          "to find topics from."}

    model = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=400)
    doc_topic = model.fit_transform(X)   # (n_docs, n_topics)
    terms = vectorizer.get_feature_names_out()

    topics = []
    for topic_idx, topic_weights in enumerate(model.components_):
        top_indices = topic_weights.argsort()[::-1][:n_top_words]
        keywords = [terms[i] for i in top_indices if topic_weights[i] > 0]
        if not keywords:
            continue
        # Documents where this topic is the dominant one, ranked by strength.
        doc_scores = doc_topic[:, topic_idx]
        dominant = [i for i in range(len(texts)) if doc_topic[i].argmax() == topic_idx]
        dominant.sort(key=lambda i: doc_scores[i], reverse=True)
        topics.append({
            "id": topic_idx,
            "keywords": keywords,
            "doc_indices": dominant,
            "weight": float(doc_scores[dominant].sum()) if dominant else 0.0,
        })

    # Drop topics with no dominant documents -- these can occur when NMF's
    # argmax never picks this component for any input, and would otherwise
    # render as a keyword card claiming "0 question(s) match this theme",
    # which is confusing rather than useful on an admin dashboard.
    topics = [t for t in topics if t["doc_indices"]]

    topics.sort(key=lambda t: len(t["doc_indices"]), reverse=True)
    return {"ok": True, "topics": topics}