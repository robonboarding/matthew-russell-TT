"""
Central configuration.

JUSTIFICATION: A single config module keeps all magic numbers and model
names in one place. During the assessment, if the brief specifies a
different embedding model or chunk size, you change one file. This is
also where production would load from Azure Key Vault.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
INDEX_PATH = DATA_PROCESSED / "faiss.index"
CHUNKS_PATH = DATA_PROCESSED / "chunks.json"
EVAL_PATH = ROOT / "eval" / "qa_pairs.json"
LOG_PATH = ROOT / "data" / "processed" / "audit_log.jsonl"


@dataclass(frozen=True)
class Config:
        azure_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
        azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")                                                                                                                                                         
        azure_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")                                                                                                                                 
        chat_deployment: str = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini")                                                                                                                                             
        embedding_deployment: str = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")                                                                                                                        
        embedding_api_version: str = os.getenv("AZURE_EMBEDDING_API_VERSION", "2023-05-15")           
        generation_model: str = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
        chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
        chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
        top_k: int = int(os.getenv("TOP_K", "5"))
        retrieve_k: int = int(os.getenv("RETRIEVE_K", "10"))
        temperature: float = 0.0  # deterministic for grounded QA


CONFIG = Config()
