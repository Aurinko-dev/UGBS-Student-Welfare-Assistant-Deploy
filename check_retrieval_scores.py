"""
Run this once to see the REAL distance scores your machine's embeddings
produce, so we can set CONFIDENCE_THRESHOLD in app.py to a number that
actually matches good results instead of rejecting them.

    python check_retrieval_scores.py
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import config

embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
vectorstore = Chroma(persist_directory=config.DB_DIR, embedding_function=embeddings)

test_queries = [
    "What are the eligibility requirements for SFAO financial aid?",   # known to work
    "I am failing two of my courses and worried about probation",     # known to fail
    "I can't pay my fees and I'm failing two courses",                # known to fail
    "How do I book a counselling appointment?",
    # Added after these were all wrongly rejected in the live app --
    # queries taken directly from real screenshots of the running assistant.
    "I've been feeling overwhelmed and anxious about school.",
    "Who is my course advisor?",
    "Which department runs the Marketing option?",
    "what of analytics option",                       # a real follow-up with no context of its own
    "What options can I choose at UGBS?",
    "What can I do about a problem with my roommate?",
]

for q in test_queries:
    print(f"\nQuery: {q}")
    results = vectorstore.similarity_search_with_score(q, k=3)
    for doc, score in results:
        source = doc.metadata.get("source", "?")
        preview = doc.page_content[:70].replace("\n", " ")
        print(f"  distance={score:.4f}  source={source}  |  {preview}...")