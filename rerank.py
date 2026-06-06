"""Module 8 — Thursday Stretch (Honors Track): Cross-Encoder Re-Ranking.

Implement a two-stage retrieval pipeline:
1. Candidate generation: Hybrid search (BM25 + Dense) to get top-k_in results.
2. Re-ranking: Cross-encoder to score (query, candidate) pairs and return top-k_out.
"""

from __future__ import annotations

import weaviate
from sentence_transformers import CrossEncoder

# The Cross-Encoder model specified in the assignment
# Initializing at module scope to avoid re-loading per call
_CE_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def cross_encoder_rerank(query: str, candidates: list[dict], k_out: int = 5) -> list[str]:
    """Score (query, candidate.text) pairs using a Cross-Encoder.

    Args:
        query: The user's search query.
        candidates: List of dicts, each containing at minimum 'doc_id' and 'text'.
        k_out: Number of top candidates to return after re-ranking.

    Returns:
        list[str]: The top-k_out doc_id strings, ordered by Cross-Encoder score descending.
    """
    if not candidates:
        return []

    # Build (query, text) pairs for the Cross-Encoder
    # We use candidate["text"] as the content to compare against the query
    pairs = [[query, c["text"]] for c in candidates]

    # Predict scores for all pairs
    # Cross-encoders process the query and document jointly for higher accuracy
    scores = _CE_MODEL.predict(pairs)

    # Sort candidates by score descending and return top k_out doc_ids
    # Zip scores with doc_ids to maintain the mapping during sort
    scored_results = sorted(
        zip(scores, [c["doc_id"] for c in candidates]),
        key=lambda x: x[0],
        reverse=True
    )

    return [res[1] for res in scored_results[:k_out]]


def rerank_search(
    client: weaviate.Client,
    query: str,
    embedder,
    k_in: int = 50,
    k_out: int = 5
) -> list[str]:
    """Two-stage search: Hybrid candidate generation followed by re-ranking.

    Args:
        client: Weaviate client.
        query: Search query string.
        embedder: The dense embedder (Bi-Encoder) used for the first stage.
        k_in: Number of candidates to retrieve in the first stage (default 50).
        k_out: Number of candidates to return after re-ranking (default 5).

    Returns:
        list[str]: The final top-k_out doc_ids.
    """
    # Stage 1 — Hybrid retrieve k_in candidates.
    # We need both 'doc_id' for identification and 'text' for the Cross-Encoder stage.
    qv = embedder.encode(query).tolist()
    
    response = (
        client.query.get("Post", ["doc_id", "text"])
        .with_hybrid(query=query, vector=qv, alpha=0.5)
        .with_limit(k_in)
        .do()
    )

    # Extracting results safely from Weaviate response
    if "data" not in response or not response["data"]["Get"]["Post"]:
        return []
    
    candidates = response["data"]["Get"]["Post"]

    # Stage 2 — Cross-encoder re-rank to k_out.
    return cross_encoder_rerank(query, candidates, k_out)