from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import chromadb
from llama_index.core import Settings as LlamaSettings, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.core.config import get_settings
from src.core.logging import get_logger
from src.rag.indexer import _get_chroma_collection, _get_embed_model

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)


def _reciprocal_rank_fusion(
    results_a: list[RetrievedChunk],
    results_b: list[RetrievedChunk],
    weight_a: float,
    weight_b: float,
    k: int = 60,
) -> list[RetrievedChunk]:
    """Combine two ranked lists via Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(results_a):
        key = chunk.text[:100]
        scores[key] = scores.get(key, 0.0) + weight_a / (k + rank + 1)
        chunks[key] = chunk

    for rank, chunk in enumerate(results_b):
        key = chunk.text[:100]
        scores[key] = scores.get(key, 0.0) + weight_b / (k + rank + 1)
        if key not in chunks:
            chunks[key] = chunk

    sorted_keys = sorted(scores, key=lambda k_: scores[k_], reverse=True)
    fused = []
    for key in sorted_keys:
        c = chunks[key]
        fused.append(
            RetrievedChunk(
                text=c.text,
                source=c.source,
                score=round(scores[key], 4),
                metadata=c.metadata,
            )
        )
    return fused


def _vector_search(query: str, top_k: int) -> list[RetrievedChunk]:
    cfg = get_settings()
    LlamaSettings.embed_model = _get_embed_model()
    LlamaSettings.llm = None

    collection = _get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    results = []
    for n in nodes:
        results.append(
            RetrievedChunk(
                text=n.get_content(),
                source=n.metadata.get("source", "unknown"),
                score=float(n.score or 0.0),
                metadata=n.metadata,
            )
        )
    return results


def _bm25_search(query: str, top_k: int) -> list[RetrievedChunk]:
    """Keyword search using BM25 over documents stored in ChromaDB."""
    collection = _get_chroma_collection()

    results = collection.get(
        include=["documents", "metadatas"]
    )

    docs = results.get("documents") or []
    metas = results.get("metadatas") or []

    if not docs:
        return []

    # Simple tokenization
    tokenized_docs = [
        doc.lower().split()
        for doc in docs
    ]

    tokenized_query = query.lower().split()

    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    chunks = []

    for idx in ranked_indices:
        chunks.append(
            RetrievedChunk(
                text=docs[idx],
                source=(metas[idx] or {}).get("source", "unknown"),
                score=float(scores[idx]),
                metadata=metas[idx] or {},
            )
        )

    return chunks


def hybrid_search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Combine vector + keyword search via Reciprocal Rank Fusion."""
    cfg = get_settings()
    k = top_k or cfg.top_k_retrieval

    if get_document_count() == 0:
        logger.warning("Vector store is empty — no documents indexed yet")
        return []

    vector_results = _vector_search(query, k)
    bm25_results = _bm25_search(query, k)

    fused = _reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        weight_a=cfg.vector_weight,
        weight_b=cfg.bm25_weight,
    )
    return fused[:k]


def get_document_count() -> int:
    return _get_chroma_collection().count()
