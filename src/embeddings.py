"""
embeddings.py -- Embedding Generation
-------------------------------------
Converts text chunks into vector embeddings using Azure OpenAI.

Model: text-embedding-3-large (recommended for high accuracy)
    - 3072 dimensions
    - State-of-the-art semantic retrieval quality
    - Uses Azure OpenAI API for consistency with LLM backend
"""

import os
import logging

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

logger = logging.getLogger(__name__)

# -- Configuration (reuses existing Azure OpenAI setup) ---------------------

load_dotenv(os.path.join("..", ".env"))

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Embedding model deployment name in Azure (may differ from LLM deployment)
# Default to text-embedding-3-large if not specified
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")


def get_embedding_function() -> AzureOpenAIEmbeddings:
    """
    Create and return a LangChain-compatible embedding function using Azure OpenAI.

    Uses text-embedding-3-large for high-quality semantic embeddings.
    Reuses the same Azure OpenAI credentials as the LLM for consistency.

    Returns:
        AzureOpenAIEmbeddings instance ready for use.
    
    Raises:
        EnvironmentError: If Azure OpenAI credentials are not configured.
    """
    if not all([AZURE_ENDPOINT, AZURE_API_VERSION, AZURE_API_KEY]):
        raise EnvironmentError(
            "\n\n"
            "===========================================================\n"
            "  AZURE OPENAI CONFIGURATION NOT COMPLETE!\n"
            "===========================================================\n"
            "  Embedding function requires Azure OpenAI credentials.\n"
            "  Ensure these are set in .env:\n"
            "     AZURE_OPENAI_ENDPOINT=https://...\n"
            "     AZURE_OPENAI_API_VERSION=2025-04-01-preview\n"
            "     AZURE_OPENAI_API_KEY=sk-...\n"
            "  Optional (defaults to text-embedding-3-large):\n"
            "     AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large\n"
            "  Get keys from: https://portal.azure.com/\n"
            "===========================================================\n"
        )

    logger.info(f"Loading Azure OpenAI embeddings: {AZURE_EMBEDDING_DEPLOYMENT}")
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
        api_key=AZURE_API_KEY,
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
        model="text-embedding-3-large",
    )
    logger.info("Azure OpenAI embedding model loaded successfully")
    return embeddings
