"""
Graph-RAG over the CrisisMMD disaster knowledge graph.

A hybrid retrieval-augmented-generation system that answers natural-language
questions grounded in the KG built by `pipeline_kg.py`:

    question
      -> query understanding   (entities + typed graph filters)
      -> hybrid retrieval       (structured graph query + vector + k-hop expansion)
      -> context assembly       (tweets + triples + provenance)
      -> generation             (local HF instruct model, cited answer)

Modules
-------
kg_store           load the KG and turn Tweet nodes into retrieval documents
embed              sentence-transformer embeddings + FAISS index
query_understanding parse a question into entities and typed filters
retriever          hybrid retriever (structured + semantic + graph expansion + RRF)
generator          local HuggingFace instruct-LLM wrapper
"""

from .kg_store import KGStore, TweetDoc

__all__ = ["KGStore", "TweetDoc"]
