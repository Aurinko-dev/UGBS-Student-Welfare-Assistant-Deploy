"""
Sentiment analysis: scores how distressed a message *sounds*, independent of
whether it contains any of risk_classifier.py's specific keyword or phrase
triggers. Feeds into the same escalation policy as an additional, orthogonal
safety net -- a message can now be flagged for human review purely on tone
(e.g. "I am really struggling and don't know what to do anymore") even when
it doesn't use any of the crisis/possible-crisis/urgency wording the rest of
the pipeline looks for.

Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) rather than a
trained/fine-tuned model: it's a lexicon + rule-based sentiment tool built
for short, informal text (tuned on social-media-style writing), which is a
good match for how students actually type in a chat box. It needs no
training data of its own -- important here, since this project's 102-row
labeled set has no sentiment labels at all -- and it adds no extra model
download or GPU/CPU inference cost on top of the embedding model and
classifier net this app already loads.

This is a signal, not a diagnosis. It does not replace, override, or rank
above the deterministic crisis phrase list, the harassment safety net, or
the urgency floor -- it only adds one more independent path to escalation
for messages that would otherwise slip through as "just" Low/Medium.
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# Below this, a negative-leaning message is treated as ordinary venting/
# frustration, not distress worth flagging. Chosen by inspecting VADER's
# compound score on example messages during development (see the project
# report for the worked examples) -- genuinely distressed, help-seeking
# phrasing ("I am really struggling and don't know what to do anymore")
# scored in the -0.40 to -0.75 range, while mild negativity ("my roommate is
# annoying but it's fine") stayed near zero or positive. NOT validated
# against a labeled distress dataset (none exists for this project) -- this
# is a heuristic cutoff, same status as risk_classifier.py's urgency-floor
# regexes, and should be revisited if it proves too sensitive/insensitive
# once real usage data accumulates.
DISTRESS_THRESHOLD = 0.4


def distress_score(text: str) -> float:
    """Returns a 0.0-1.0 distress score for a message: 0.0 = neutral or
    positive, 1.0 = maximally negative. This is just VADER's compound score
    (-1..+1) rescaled to keep only the negative half, since positivity isn't
    meaningful for this use case -- a very upbeat message isn't "less
    escalatable" than a neutral one, so both map near 0."""
    if not text or not text.strip():
        return 0.0
    compound = _analyzer.polarity_scores(text)["compound"]
    return max(0.0, -compound)


def is_high_distress(text: str, threshold: float = DISTRESS_THRESHOLD) -> bool:
    """True when the message's tone alone justifies a closer look, per
    DISTRESS_THRESHOLD. See distress_score() and the module docstring for
    what this does and doesn't mean."""
    return distress_score(text) >= threshold