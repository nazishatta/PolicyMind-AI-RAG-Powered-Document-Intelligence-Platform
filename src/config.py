"""Configuration for the PolicyMind AI Streamlit layer — loaded from .env via python-dotenv."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Directories ---
RAW_DATA_DIR: str = os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR: str = os.getenv("PROCESSED_DATA_DIR", "data/processed")
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "vector_db/chroma")

# --- Embedding ---
EMBEDDING_MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# --- Chunking ---
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Retrieval ---
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

# --- LLM (optional) ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Ensure directories exist at import time so callers never have to mkdir.
for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, CHROMA_DB_PATH):
    Path(_dir).mkdir(parents=True, exist_ok=True)
