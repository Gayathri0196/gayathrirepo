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
    Build a Chroma vector store from ingested chunks (text and images).

    Each chunk becomes a Document with:
    - page_content: the section text or image analysis
    - metadata: domain, section_id, section_title, source_doc, page_range, and optional image data

    Args:
        chunks: List of chunk dicts from ingestion.py (may include image chunks)

    Returns:
        Chroma vector store instance (already persisted to disk)
    """
    logger.info(f"Building vector store with {len(chunks)} chunks (including images)...")

    # Convert chunks to LangChain Document objects
    documents = []
    text_count = 0
    image_count = 0
    
    for chunk in chunks:
        # Build base metadata
        metadata = {
            "domain": chunk["domain"],
            "section_id": chunk["section_id"],
            "section_title": chunk["section_title"],
            "source_doc": chunk["source_doc"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
        }
        
        # Add image-specific metadata if present
        if "image_path" in chunk:
            metadata["image_path"] = chunk["image_path"]
            metadata["is_image_chunk"] = True
            if "image_metadata" in chunk:
                # Store essential image metadata
                img_meta = chunk["image_metadata"]
                metadata["image_page"] = img_meta.get("page_number")
                metadata["image_index"] = img_meta.get("image_index")
            image_count += 1
        else:
            metadata["is_image_chunk"] = False
            text_count += 1
        
        # Add analysis metadata if present
        if "analysis_metadata" in chunk:
            metadata["ocr_confidence"] = chunk["analysis_metadata"].get("ocr_confidence", 0)
            metadata["has_azure_analysis"] = chunk["analysis_metadata"].get("has_azure_analysis", False)
        
        doc = Document(
            page_content=chunk["text"],
            metadata=metadata,
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
        f"({len(documents)} documents: {text_count} text, {image_count} images)"
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
