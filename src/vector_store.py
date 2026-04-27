"""
vector_store.py -- Chroma Vector Database
-----------------------------------------
Stores chunk embeddings in a persistent Chroma database.

DATABASE LOCATION: data/vectordb/
  - This is a local folder-based database (no server needed).
  - Chroma stores embeddings + metadata as files in this folder.
  - Survives restarts -- data persists between runs.
  - To reset: just delete the data/vectordb/ folder.

HOW IT WORKS:
  1. Takes chunks from ingestion.py (text + metadata)
  2. Uses embeddings.py to convert text -> vectors
  3. Stores vectors + metadata in Chroma
  4. Later, retriever.py searches this database
"""

import os
import logging
from langchain_chroma import Chroma
from langchain_core.documents import Document

from embeddings import get_embedding_function

logger = logging.getLogger(__name__)

# -- Configuration -------------------------------------------------------------

# Where Chroma stores its database files on disk
VECTORDB_DIR = os.path.join("..", "data", "vectordb")

# Collection name inside Chroma (like a "table name" in a regular database)
COLLECTION_NAME = "lms_validation_docs"


def build_vector_store(chunks: list) -> Chroma:
    """
    Build a Chroma vector store from ingested chunks.

    Each chunk becomes a Document with:
    - page_content: the section text
    - metadata: domain, section_id, section_title, source_doc, page_range

    Args:
        chunks: List of chunk dicts from ingestion.py

    Returns:
        Chroma vector store instance (already persisted to disk)
    """
    logger.info(f"Building vector store with {len(chunks)} chunks...")

    # Convert chunks to LangChain Document objects
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "domain": chunk["domain"],
                "section_id": chunk["section_id"],
                "section_title": chunk["section_title"],
                "source_doc": chunk["source_doc"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
            },
        )
        documents.append(doc)

    # Get embedding function
    embedding_fn = get_embedding_function()

    # Delete existing collection to prevent duplicate entries on re-ingest
    import chromadb
    chroma_client = chromadb.PersistentClient(path=VECTORDB_DIR)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted existing collection '{COLLECTION_NAME}' to prevent duplicates")
    except ValueError:
        pass  # Collection doesn't exist yet

    # Create Chroma and persist to disk
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_fn,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORDB_DIR,
    )

    logger.info(
        f"Vector store built and persisted at '{VECTORDB_DIR}' "
        f"({len(documents)} documents in collection '{COLLECTION_NAME}')"
    )
    return vectordb


def load_vector_store() -> Chroma:
    """
    Load an existing Chroma vector store from disk.

    Call this when the database has already been built (by build_vector_store).
    Avoids re-embedding all chunks on every run.

    Returns:
        Chroma vector store instance loaded from disk.
    """
    if not os.path.exists(VECTORDB_DIR):
        raise FileNotFoundError(
            f"Vector database not found at '{VECTORDB_DIR}'. "
            "Run the ingestion + build pipeline first."
        )

    embedding_fn = get_embedding_function()

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=VECTORDB_DIR,
    )

    logger.info(f"Loaded existing vector store from '{VECTORDB_DIR}'")
    return vectordb
