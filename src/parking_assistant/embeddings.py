"""Embedding providers.

``fastembed`` (default) runs a local ONNX sentence-embedding model.
``hashing`` is a deterministic, dependency-free bag-of-words embedding used for
offline runs and tests; it captures word overlap, not meaning.
"""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

from parking_assistant.config import Settings

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it my of on or "
    "the to what when where which who will with you your me there this that "
    "we our us".split()
)


def _stem(token: str) -> str:
    for suffix in ("ing", "ies", "est", "es", "ed", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            if len(token) > 3 and token[-1] == token[-2] and token[-1] not in "aeiou":
                token = token[:-1]
            return token
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class HashingEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = tokenize(text)
        features = tokens + [
            f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)
        ]
        for feature in features:
            digest = hashlib.md5(feature.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.embeddings_provider == "hashing":
        return HashingEmbeddings()
    return FastEmbedEmbeddings(
        settings.fastembed_model, cache_dir=str(settings.data_dir / "fastembed")
    )


def describe_embeddings(settings: Settings) -> str:
    if settings.embeddings_provider == "hashing":
        return "hashing-bow-512"
    return f"fastembed:{settings.fastembed_model}"
