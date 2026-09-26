"""
Builds the ChromaDB vector store from the markdown knowledge base.
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
from langchain_chroma import Chroma

import config

DB_DIR = config.DB_DIR
MARKDOWN_FILES = config.MARKDOWN_FILES

HEADERS_TO_SPLIT_ON = [
    ("#", "title"),
    ("##", "section"),
    ("###", "entry"),
]

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
        strip_headers=False,
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
            hc.metadata.update(doc.metadata)
            if len(hc.page_content) <= FALLBACK_CHUNK_SIZE:
                chunks.append(hc)
            else:
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
    if os.path.exists(DB_DIR):
        print(f"Removing existing vector store at '"'"'{DB_DIR}'"'"' before rebuilding...")
        for attempt in range(5):
            try:
                shutil.rmtree(DB_DIR)
                break
            except PermissionError:
                if attempt == 4:
                    sys.exit(
                        f"\nCould not delete '"'"'{DB_DIR}'"'"' -- it'"'"'s locked by another "
                        "process.\nClose any other terminal or Streamlit app that "
                        "has this project open, then run this script again.\n"
                        f"If that doesn'"'"'t help, delete the '"'"'{DB_DIR}'"'"' folder by "
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
    print(f"Successfully saved vector DB to '"'"'{DB_DIR}'"'"'!")
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
