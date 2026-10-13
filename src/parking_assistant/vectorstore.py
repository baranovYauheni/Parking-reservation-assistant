"""Vector storage for public knowledge chunks (Weaviate or in-memory)."""

from __future__ import annotations

import logging
from typing import Protocol

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore

from parking_assistant.config import Settings

logger = logging.getLogger(__name__)

METADATA_KEYS = ("doc_id", "title", "section", "source", "chunk_id")


class KnowledgeIndex(Protocol):
    def rebuild(self, documents: list[Document]) -> None: ...

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Return documents with cosine similarity (higher is better)."""
        ...

    def close(self) -> None: ...


class MemoryIndex:
    def __init__(self, embeddings: Embeddings) -> None:
        self.embeddings = embeddings
        self.store = InMemoryVectorStore(embeddings)

    def rebuild(self, documents: list[Document]) -> None:
        self.store = InMemoryVectorStore(self.embeddings)
        self.store.add_documents(documents)

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        return self.store.similarity_search_with_score(query, k=k)

    def close(self) -> None:
        pass


class WeaviateIndex:
    """Weaviate collection with self-provided vectors and cosine distance."""

    def __init__(self, settings: Settings, embeddings: Embeddings) -> None:
        import weaviate

        self.embeddings = embeddings
        self.name = settings.weaviate_collection
        if settings.weaviate_mode == "embedded":
            self.client = weaviate.connect_to_embedded(
                persistence_data_path=str(settings.weaviate_data_path),
                port=settings.weaviate_port,
                grpc_port=settings.weaviate_grpc_port,
                environment_variables={
                    "CLUSTER_ADVERTISE_ADDR": "127.0.0.1",
                    "LOG_LEVEL": "warning",
                    "DISABLE_TELEMETRY": "true",
                },
            )
        else:
            self.client = weaviate.connect_to_local(
                host=settings.weaviate_host,
                port=settings.weaviate_port,
                grpc_port=settings.weaviate_grpc_port,
            )

    def rebuild(self, documents: list[Document]) -> None:
        from weaviate.classes.config import (
            Configure,
            DataType,
            Property,
            VectorDistances,
        )
        from weaviate.classes.data import DataObject

        if self.client.collections.exists(self.name):
            self.client.collections.delete(self.name)
        collection = self.client.collections.create(
            self.name,
            properties=[Property(name="text", data_type=DataType.TEXT)]
            + [Property(name=key, data_type=DataType.TEXT) for key in METADATA_KEYS],
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                )
            ),
        )
        vectors = self.embeddings.embed_documents([d.page_content for d in documents])
        objects = [
            DataObject(
                properties={"text": doc.page_content}
                | {key: str(doc.metadata.get(key, "")) for key in METADATA_KEYS},
                vector=vector,
            )
            for doc, vector in zip(documents, vectors, strict=True)
        ]
        result = collection.data.insert_many(objects)
        if result.has_errors:
            raise RuntimeError(f"Weaviate insert failed: {result.errors}")

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        from weaviate.classes.query import MetadataQuery

        collection = self.client.collections.get(self.name)
        response = collection.query.near_vector(
            near_vector=self.embeddings.embed_query(query),
            limit=k,
            return_metadata=MetadataQuery(distance=True),
        )
        results = []
        for obj in response.objects:
            props = dict(obj.properties)
            text = props.pop("text")
            similarity = 1.0 - float(obj.metadata.distance or 0.0)
            results.append((Document(page_content=text, metadata=props), similarity))
        return results

    def close(self) -> None:
        self.client.close()


def build_index(settings: Settings, embeddings: Embeddings) -> KnowledgeIndex:
    if settings.vector_backend == "memory":
        return MemoryIndex(embeddings)
    return WeaviateIndex(settings, embeddings)


class KnowledgeRetriever(BaseRetriever):
    """LangChain retriever that drops results below a similarity threshold."""

    index: object
    k: int = 3
    min_score: float = 0.0

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        docs = []
        for doc, score in self.index.search(query, self.k):
            if score >= self.min_score:
                doc.metadata["score"] = round(score, 4)
                docs.append(doc)
        return docs
