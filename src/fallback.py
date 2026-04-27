"""
fallback.py -- Fallback Handler for No-Context Scenarios
--------------------------------------------------------
When the retriever finds no relevant chunks (all below the similarity
threshold), this module provides the "I don't know" response.

Key design: We check BEFORE calling the LLM, so we don't waste
API calls on questions without supporting context.
"""

import logging

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = (
    "I don't know. The provided documents don't contain sufficient "
    "information to answer this question. Please try asking about "
    "topics covered in the LMS Test Plan, such as:\n"
    "  - Testing approach (ST / UAT)\n"
    "  - Test environment\n"
    "  - Test data strategy\n"
    "  - Test tools\n"
    "  - Defect management\n"
    "  - Traceability\n"
    "  - Roles and responsibilities"
)


def check_fallback(retrieved_docs: list) -> bool:
    """
    Check if the retrieval returned any relevant documents.

    Args:
        retrieved_docs: List of documents from the retriever.

    Returns:
        True if we should use fallback (no docs found).
        False if there are relevant docs to use.
    """
    if not retrieved_docs:
        logger.info("FALLBACK triggered: No relevant documents found.")
        return True
    return False


def get_fallback_response() -> dict:
    """
    Return the standard fallback response.

    Returns:
        Dict with answer and metadata.
    """
    return {
        "answer": FALLBACK_MESSAGE,
        "source_documents": [],
        "fallback": True,
    }
