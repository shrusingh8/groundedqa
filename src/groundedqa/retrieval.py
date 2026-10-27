"""Chunking and brute-force cosine search. ponytail: numpy scan, swap in FAISS/Chroma past ~100k chunks."""
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .llm import Embed


@dataclass
class Chunk:
    doc: str
    text: str


def chunk_text(text: str, size: int = 120, overlap: int = 30) -> List[str]:
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def load_chunks(folder: str, size: int = 120, overlap: int = 30) -> List[Chunk]:
    return [Chunk(p.name, t) for p in sorted(Path(folder).glob("*.txt"))
            for t in chunk_text(p.read_text(encoding="utf-8"), size, overlap)]


def tfidf_embed(corpus: List[str]) -> Embed:
    """Offline embedder for tests and keyless demos; use `llm.openai_embed` for real semantic search."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(stop_words="english").fit(corpus)
    return lambda texts: vec.transform(texts).toarray()


class Index:
    def __init__(self, chunks: List[Chunk], embed: Embed):
        self.chunks, self.embed = chunks, embed
        self.matrix = self._unit(embed([c.text for c in chunks]))

    @staticmethod
    def _unit(m: np.ndarray) -> np.ndarray:
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

    def search(self, query: str, k: int = 4) -> List[Chunk]:
        scores = self.matrix @ self._unit(self.embed([query]))[0]
        return [self.chunks[i] for i in np.argsort(-scores)[:k]]
