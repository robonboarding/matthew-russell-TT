"""
Embedding generation and vector index construction.

JUSTIFICATION SUMMARY:
- OpenAI text-embedding-3-small chosen for speed of setup and low cost
- FAISS flat L2 index for simplicity on small corpora
- Batching embeddings to respect rate limits and reduce API calls
- Index and metadata stored together so retrieval is reproducible

In production at Rabobank:
- Switch to Azure OpenAI embeddings for data residency
- For larger corpora use FAISS IVF or HNSW, or move to Azure AI Search / Neo4j
- Add re-embedding pipeline when documents update
"""
import os 
import pickle
from pathlib import Path

import faiss
import numpy as np
from openai import AzureOpenAI, api_key

from src.config import CONFIG, DATA_PROCESSED, INDEX_PATH
from src.ingest import Chunk, load_chunks

CLIENT = AzureOpenAI(                                                                                                                                                                                                    
      api_key=CONFIG.azure_api_key,
      api_version=CONFIG.azure_api_version,                                                                                                                                                                                
      azure_endpoint=CONFIG.azure_endpoint,                                                                                                                                                                                
  )
METADATA_PATH = DATA_PROCESSED / "chunk_metadata.pkl"


def embed_batch(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """
    Embed texts in batches.

    JUSTIFICATION: OpenAI accepts up to ~2000 inputs per call but batching
    smaller keeps memory low and lets us recover from a single failed batch
    without re-embedding everything. For production, add retry with
    exponential backoff and dead-letter queue for persistent failures.
    """
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = CLIENT.embeddings.create(
            model= CONFIG.embedding_deployment,
            input=batch,
        )
        all_embeddings.extend(e.embedding for e in response.data)
        print(f"Embedded {min(i + batch_size, len(texts))} / {len(texts)}")
    return np.array(all_embeddings, dtype=np.float32)


def build_index(chunks: list[Chunk]) -> None:
    """Build a FAISS index over chunk embeddings and persist to disk."""
    texts = [c.text for c in chunks]
    embeddings = embed_batch(texts)

    # Flat L2 index. For larger corpora swap to IndexIVFFlat or IndexHNSWFlat.
    # JUSTIFICATION: flat index is exact and fast up to ~100k vectors.
    # Above that, approximate indexes trade a tiny recall loss for huge speed gains.
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    METADATA_PATH.write_bytes(pickle.dumps(chunks))
    print(f"Built index with {index.ntotal} vectors (dim {dim})")


def load_index() -> tuple[faiss.Index, list[Chunk]]:
    """Load the persisted index and chunk metadata."""
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Index not built. Run `python -m src.ingest` then `python -m src.index`."
        )
    index = faiss.read_index(str(INDEX_PATH))
    chunks = pickle.loads(METADATA_PATH.read_bytes())
    return index, chunks


if __name__ == "__main__":
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("No chunks found. Run `python -m src.ingest` first.")
    build_index(chunks)
