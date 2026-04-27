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
from typing import List

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from openai import AzureOpenAI

from token_tracker import record_usage

logger = logging.getLogger(__name__)

# -- Configuration (reuses existing Azure OpenAI setup) ---------------------

load_dotenv(os.path.join("..", ".env"))

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Embedding model deployment name in Azure (may differ from LLM deployment)
# Default to text-embedding-3-large if not specified
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")


class TrackedAzureEmbeddings(Embeddings):
    """Embedding wrapper that records token usage and estimated cost."""

    def __init__(self, endpoint: str, api_version: str, api_key: str, deployment: str):
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=api_key,
        )
        self.deployment = deployment

    def _embed(self, texts: List[str], operation: str) -> List[List[float]]:
        inputs = [t if isinstance(t, str) else str(t) for t in texts]
        if not inputs:
            return []

        response = self.client.embeddings.create(
            model=self.deployment,
            input=inputs,
        )

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens) if usage else prompt_tokens
        input_tokens = total_tokens or prompt_tokens or 0

        record_usage(
            operation=operation,
            model=self.deployment,
            input_tokens=input_tokens,
            output_tokens=0,
            extra={"items": len(inputs)},
        )

        return [item.embedding for item in response.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, operation="embedding_documents")

    def embed_query(self, text: str) -> List[float]:
        vectors = self._embed([text], operation="embedding_query")
        return vectors[0] if vectors else []


def get_embedding_function() -> Embeddings:
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
    embeddings = TrackedAzureEmbeddings(
        endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
        api_key=AZURE_API_KEY,
        deployment=AZURE_EMBEDDING_DEPLOYMENT,
    )
    logger.info("Azure OpenAI embedding model loaded successfully")
    return embeddings
