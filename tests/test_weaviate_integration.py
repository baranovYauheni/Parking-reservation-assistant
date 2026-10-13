"""Integration test against a real (embedded or local) Weaviate.

Run with: RUN_WEAVIATE_TESTS=1 uv run pytest -m integration
"""

import os

import pytest

from parking_assistant.config import Settings
from parking_assistant.embeddings import HashingEmbeddings
from parking_assistant.knowledge import load_chunks
from parking_assistant.vectorstore import WeaviateIndex
from tests.conftest import KNOWLEDGE_DIR

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("RUN_WEAVIATE_TESTS"), reason="set RUN_WEAVIATE_TESTS=1"
    ),
]


@pytest.fixture
def index(tmp_path):
    settings = Settings(
        PARKING_DATA_DIR=tmp_path,
        WEAVIATE_PORT=8089,
        WEAVIATE_GRPC_PORT=50089,
        WEAVIATE_COLLECTION="ParkingKnowledgeTest",
        _env_file=None,
    )
    idx = WeaviateIndex(settings, HashingEmbeddings())
    yield idx
    idx.close()


def test_weaviate_rebuild_and_search_returns_metadata(index):
    index.rebuild(load_chunks(KNOWLEDGE_DIR))
    results = index.search("maximum vehicle height", 3)
    assert results[0][0].metadata["doc_id"] == "rules"
    assert 0 < results[0][1] <= 1


def test_weaviate_rebuild_is_idempotent(index):
    chunks = load_chunks(KNOWLEDGE_DIR)
    index.rebuild(chunks)
    index.rebuild(chunks)
    collection = index.client.collections.get(index.name)
    assert collection.aggregate.over_all(total_count=True).total_count == len(chunks)
