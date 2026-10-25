from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from resume_search.derive.models import CandidateRecord

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384


class Embedder:
    """Loads the model on first use; one instance per process is enough."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        out = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(out, dtype=np.float32)


_default = Embedder()


def default_embedder() -> Embedder:
    return _default


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype="<f4").tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4")


def candidate_vectors(
    record: CandidateRecord, embedder: Embedder | None = None
) -> list[tuple[str, int, np.ndarray]]:
    """One vector for the search summary and one per experience entry. Neither text carries
    contact data (the summary validator forbids it; bullets are job content)."""
    ex = record.extracted
    texts = [record.derived.search_summary] + [
        f"{e.title} at {e.employer}. " + " ".join(e.bullets) for e in ex.experience
    ]
    vecs = (embedder or _default).embed(texts)
    keys = [("summary", 0)] + [("role", i) for i in range(len(ex.experience))]
    return [(kind, idx, v) for (kind, idx), v in zip(keys, vecs, strict=True)]


@dataclass(frozen=True)
class VectorHit:
    candidate_id: str
    score: float
    kind: str
    idx: int


class VectorIndex:
    def __init__(self, keys: list[tuple[str, str, int]], matrix: np.ndarray):
        self.keys = keys
        self.matrix = matrix

    @classmethod
    def load(cls, conn: sqlite3.Connection) -> VectorIndex:
        rows = conn.execute(
            "SELECT candidate_id, kind, idx, vec FROM embeddings ORDER BY candidate_id, kind, idx"
        ).fetchall()
        keys = [(r["candidate_id"], r["kind"], r["idx"]) for r in rows]
        matrix = (
            np.stack([from_blob(r["vec"]) for r in rows])
            if rows
            else np.zeros((0, DIM), dtype=np.float32)
        )
        return cls(keys, matrix)

    def scores(self, query_vec: np.ndarray) -> dict[str, float]:
        """Cosine (vectors are normalized, so a dot product), best vector per candidate."""
        if not self.keys:
            return {}
        sims = self.matrix @ np.asarray(query_vec, dtype=np.float32)
        best: dict[str, float] = {}
        for (cid, _, _), score in zip(self.keys, sims, strict=True):
            best[cid] = max(best.get(cid, -1.0), float(score))
        return best

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[VectorHit]:
        """Cosine (vectors are normalized, so a dot product), best vector per candidate."""
        if not self.keys:
            return []
        scores = self.matrix @ np.asarray(query_vec, dtype=np.float32)
        best: dict[str, VectorHit] = {}
        for (cid, kind, idx), score in zip(self.keys, scores, strict=True):
            if cid not in best or score > best[cid].score:
                best[cid] = VectorHit(cid, float(score), kind, idx)
        return sorted(best.values(), key=lambda h: -h.score)[:top_k]
