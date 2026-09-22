import os
from pathlib import Path
from typing import List
from loguru import logger

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = "finance_documents"

try:
    import chromadb
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
    from langchain.schema import Document
    from .embeddings import get_embeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


def ingest_document(file_path: str) -> int:
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain/ChromaDB not fully installed. Document indexing skipped.")
        return 0
    # Ingestion logic
    return 1
