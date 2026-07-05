"""
Embedding + vector index over the KG's tweets.

Each TweetDoc is embedded with a sentence-transformer (default BAAI/bge-small-en-
v1.5: 384-dim, ~130 MB, strong on short-text retrieval) and stored in a FAISS
inner-product index. Because we L2-normalise the vectors, inner product == cosine
similarity.

The index is persisted next to the KG so it is built once (CPU is fine for ~2k
tweets) and reused at query time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .kg_store import KGStore, TweetDoc

logger = logging.getLogger(__name__)

# bge models recommend a query prefix for asymmetric search.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class VectorIndex:
    """sentence-transformer embeddings + FAISS index over TweetDocs."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None              # lazy load
        self._index = None              # faiss index
        self.ids: List[str] = []        # row -> tweet_id
        self.is_bge = "bge" in model_name.lower()

    # ── model ─────────────────────────────────────────────────

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedder '%s'", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if is_query and self.is_bge:
            texts = [_BGE_QUERY_PREFIX + t for t in texts]
        emb = self.model.encode(
            texts, batch_size=64, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=len(texts) > 500,
        )
        return emb.astype("float32")

    # ── build / persist ───────────────────────────────────────

    def build(self, store: KGStore) -> None:
        import faiss
        docs = [d for d in store.all_docs() if d.text.strip()]
        self.ids = [d.tweet_id for d in docs]
        texts = [d.embedding_text() for d in docs]
        logger.info("Embedding %d tweets…", len(texts))
        emb = self._encode(texts, is_query=False)
        self._index = faiss.IndexFlatIP(emb.shape[1])
        self._index.add(emb)
        logger.info("Built FAISS index: %d vectors, dim=%d", len(self.ids), emb.shape[1])

    def save(self, out_dir: str) -> None:
        import faiss
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(out / "tweets.faiss"))
        with open(out / "tweets.ids.json", "w") as f:
            json.dump({"model": self.model_name, "ids": self.ids}, f)
        logger.info("Saved vector index -> %s", out)

    def load(self, out_dir: str) -> None:
        import faiss
        out = Path(out_dir)
        self._index = faiss.read_index(str(out / "tweets.faiss"))
        meta = json.load(open(out / "tweets.ids.json"))
        self.ids = meta["ids"]
        self.model_name = meta.get("model", self.model_name)
        self.is_bge = "bge" in self.model_name.lower()
        logger.info("Loaded vector index (%d vectors, model=%s)",
                    len(self.ids), self.model_name)

    # ── search ────────────────────────────────────────────────

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Return [(tweet_id, score)] for the top-k most similar tweets."""
        if self._index is None:
            raise RuntimeError("Index not built/loaded.")
        q = self._encode([query], is_query=True)
        scores, idx = self._index.search(q, min(k, len(self.ids)))
        out = []
        for j, s in zip(idx[0], scores[0]):
            if j < 0:
                continue
            out.append((self.ids[j], float(s)))
        return out
