"""
Retrieval with MMR reranking.

JUSTIFICATION SUMMARY:
- Query embedded with the same model used for indexing (obvious but critical)
- Over-retrieve (k=10) then rerank down to k=5 to improve precision
- MMR (Maximum Marginal Relevance) adds diversity so we don't return
  near-duplicate chunks and waste context window
- Distance scores returned alongside chunks for downstream confidence filtering

In production at Rabobank:
- Add a cross-encoder reranker for higher precision on the final top-k
- Hybrid retrieval: combine dense embeddings with BM25 for rare terms like
  product codes, BSN patterns, or internal jargon
- Query rewriting: an LLM pre-pass that rewrites ambiguous or multi-part
  questions into cleaner retrieval queries
"""

from dataclasses import dataclass

import numpy as np
from openai import AzureOpenAI

from src.config import CONFIG
from src.index import load_index
from src.ingest import Chunk

CLIENT = AzureOpenAI(                                   
      api_key=CONFIG.azure_api_key,   
      api_version=CONFIG.azure_api_version,
      azure_endpoint=CONFIG.azure_endpoint,                                                                                                                                                                                
  )


@dataclass
class RetrievalResult:
    chunk: Chunk
    distance: float
    rank: int


def embed_query(query: str) -> np.ndarray:
    response = CLIENT.embeddings.create(model=CONFIG.embedding_deployment, input=[query])
    return np.array(response.data[0].embedding, dtype=np.float32)


def mmr_rerank(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    candidate_indices: list[int],
    k: int,
    lambda_mult: float = 0.7,
) -> list[int]:
    """
    Maximum Marginal Relevance reranking.

    JUSTIFICATION: lambda=0.7 biases toward relevance while still penalising
    near-duplicates. Rabobank policy docs often repeat clauses across
    chapters, so diversity matters to avoid feeding the LLM five copies
    of the same fact.

    Algorithm: pick the most relevant first, then iteratively pick the
    candidate that maximises (lambda * relevance - (1-lambda) * max similarity to already-picked).
    """
    if not candidate_indices:
        return []

    # Cosine-style similarity on L2-normalised vectors. FAISS flat L2
    # returns squared L2 distance; convert to similarity for scoring.
    def normalise(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.where(norm == 0, 1, norm)

    q = normalise(query_vec.reshape(1, -1))[0]
    cands = normalise(candidate_vecs)
    relevance = cands @ q  # cosine similarity

    selected: list[int] = []
    remaining = list(range(len(candidate_indices)))

    # First pick: most relevant
    first = int(np.argmax(relevance))
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < k:
        best_score = -np.inf
        best_idx = remaining[0]
        for idx in remaining:
            max_sim_to_selected = max(
                float(cands[idx] @ cands[s]) for s in selected
            )
            score = lambda_mult * relevance[idx] - (1 - lambda_mult) * max_sim_to_selected
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidate_indices[i] for i in selected]


def retrieve(
    query: str,
    top_k: int | None = None,
    retrieve_k: int | None = None,
    use_mmr: bool = True,
) -> list[RetrievalResult]:
    """
    End-to-end retrieval: embed query, search index, optionally rerank with MMR.

    use_mmr=False returns the top_k nearest neighbours directly (baseline).
    use_mmr=True over-retrieves retrieve_k candidates then reranks to top_k
    using Maximum Marginal Relevance, trading a small amount of relevance
    for diversity across near-duplicate chunks.
    """
    top_k = top_k or CONFIG.top_k
    retrieve_k = retrieve_k or CONFIG.retrieve_k

    index, chunks = load_index()
    query_vec = embed_query(query)

    search_k = retrieve_k if use_mmr else top_k
    distances, indices = index.search(query_vec.reshape(1, -1), search_k)
    candidate_indices = [int(i) for i in indices[0] if i >= 0]
    candidate_distances = [float(d) for d in distances[0][: len(candidate_indices)]]

    if not candidate_indices:
        return []

    if use_mmr:
        candidate_vecs = np.vstack([index.reconstruct(i) for i in candidate_indices])
        reranked_indices = mmr_rerank(query_vec, candidate_vecs, candidate_indices, top_k)
        distance_map = dict(zip(candidate_indices, candidate_distances))
        return [
            RetrievalResult(chunk=chunks[idx], distance=distance_map[idx], rank=rank)
            for rank, idx in enumerate(reranked_indices)
        ]

    return [
        RetrievalResult(chunk=chunks[idx], distance=dist, rank=rank)
        for rank, (idx, dist) in enumerate(zip(candidate_indices, candidate_distances))
    ]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What documents are required for a mortgage?"
    results = retrieve(query)
    print(f"Query: {query}\n")
    for r in results:
        print(f"[{r.rank}] {r.chunk.chunk_id} (d={r.distance:.3f})")
        print(f"    {r.chunk.text[:200]}...\n")
