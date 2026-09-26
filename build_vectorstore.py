"""
Builds the ChromaDB vector store from the markdown knowledge base.

Two fixes from the original version:

1. File list now comes from config.MARKDOWN_FILES instead of a second,
   separately-maintained copy of the list. The old version had its own
   hardcoded MARKDOWN_FILES that silently diverged from config.py's --
   files added to config.py were never actually being indexed. There is
   now exactly one place that lists the knowledge base files.

2. Chunking is now header-aware. The old splitter used a flat
   RecursiveCharacterTextSplitter with "Q:" as a low-priority separator,
   which still cut Q&A pairs in half whenever a pair ran long (~10% of
   chunks in the academic-affairs and BHJCR files started mid-sentence --
   see the earlier chunking test). This version first splits on markdown
   headers (## sections, ### individual Q&A entries) with
   MarkdownHeaderTextSplitter, so a "### Q: ... A: ..." entry stays whole
   as long as it fits in one chunk. Only sections that are still too long
   after that first pass get recursively split further, and that
   secondary split prefers paragraph/sentence boundaries over an
   arbitrary character count.
"""
import os
import shutil
import sys
import time

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import config

DB_DIR = config.DB_DIR
MARKDOWN_FILES = config.MARKDOWN_FILES

HEADERS_TO_SPLIT_ON = [
    ("#", "title"),
    ("##", "section"),
    ("###", "entry"),
]

# Only used as a fallback for header-sections that are still too long.
FALLBACK_CHUNK_SIZE = 500
FALLBACK_CHUNK_OVERLAP = 50


def load_md_documents():
    documents = []
    for file_path in MARKDOWN_FILES:
        if os.path.exists(file_path):
            print(f"Loading {file_path}...")
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
        else:
            print(f"Warning: File '{file_path}' not found!")
    return documents


def chunk_documents(documents):
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # keep the "### Q: ..." text in the chunk itself
    )
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=FALLBACK_CHUNK_SIZE,
        chunk_overlap=FALLBACK_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        header_chunks = header_splitter.split_text(doc.page_content)
        for hc in header_chunks:
            hc.metadata.update(doc.metadata)  # keep source filename etc.
            if len(hc.page_content) <= FALLBACK_CHUNK_SIZE:
                chunks.append(hc)
            else:
                # Section (e.g. a long BHJCR article) still too big for one
                # chunk -- split further, but on sentence/paragraph breaks,
                # never mid-word.
                sub_texts = fallback_splitter.split_text(hc.page_content)
                for st in sub_texts:
                    chunks.append(type(hc)(page_content=st, metadata=hc.metadata))

    print(f"Total chunks created: {len(chunks)}")
    mid_sentence = sum(
        1 for c in chunks if c.page_content and c.page_content[0].islower()
    )
    print(f"Chunks starting mid-sentence (lower bound on split quality): {mid_sentence}")
    return chunks


def build_and_save_vectorstore(chunks):
    print("Generating free local embeddings and saving to ChromaDB...")
    # IMPORTANT: Chroma.from_documents() does not replace an existing collection
    # at DB_DIR -- it appends to it. Re-running this script without clearing the
    # old store first means every chunk gets duplicated on each rebuild (this is
    # what produced the exact-duplicate rank-1/rank-2 results in
    # check_retrieval_scores.py after the second run). Always start clean.
    if os.path.exists(DB_DIR):
        print(f"Removing existing vector store at '{DB_DIR}' before rebuilding...")
        # Windows sometimes briefly locks a just-written folder (antivirus or
        # search indexing scanning the new files, or a leftover process that
        # still has the store open, e.g. a Streamlit tab from earlier testing).
        # Retry a few times before giving up, instead of crashing on the first try.
        for attempt in range(5):
            try:
                shutil.rmtree(DB_DIR)
                break
            except PermissionError:
                if attempt == 4:
                    sys.exit(
                        f"\nCould not delete '{DB_DIR}' -- it's locked by another "
                        "process.\nClose any other terminal or Streamlit app that "
                        "has this project open, then run this script again.\n"
                        f"If that doesn't help, delete the '{DB_DIR}' folder by "
                        "hand (File Explorer or `Remove-Item -Recurse -Force "
                        f"{DB_DIR}`) and re-run."
                    )
                time.sleep(1)

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    print(f"Successfully saved vector DB to '{DB_DIR}'!")
    return vectorstore


if __name__ == "__main__":
    print("--- Starting Free Local RAG Indexing Pipeline ---")
    docs = load_md_documents()
    if not docs:
        print("Error: No documents loaded.")
        sys.exit(1)
    chunks = chunk_documents(docs)
    build_and_save_vectorstore(chunks)
    print("--- Indexing Complete ---")