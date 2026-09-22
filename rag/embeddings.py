"""
Embedding model wrapper.
Uses Ollama's nomic-embed-text model to generate vector embeddings
for storing and searching documents in ChromaDB.
"""

from langchain_ollama import OllamaEmbeddings
from loguru import logger
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


def get_embeddings() -> OllamaEmbeddings:
    """Return the configured embedding model."""
    logger.info(f"Loading embeddings model: {EMBEDDING_MODEL}")
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
