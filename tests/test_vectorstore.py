from langchain_core.documents import Document

from parking_assistant.embeddings import HashingEmbeddings
from parking_assistant.vectorstore import KnowledgeRetriever, MemoryIndex

DOCS = [
    Document(
        page_content="The maximum vehicle height is 2.1 metres.",
        metadata={"doc_id": "rules"},
    ),
    Document(
        page_content="Payment is made by card or cash at the exit.",
        metadata={"doc_id": "pricing"},
    ),
]


def test_memory_index_returns_best_match_first_with_similarity():
    index = MemoryIndex(HashingEmbeddings())
    index.rebuild(DOCS)
    results = index.search("vehicle height limit", 2)
    assert results[0][0].metadata["doc_id"] == "rules"
    assert results[0][1] > results[1][1]


def test_rebuild_replaces_previous_documents():
    index = MemoryIndex(HashingEmbeddings())
    index.rebuild(DOCS)
    index.rebuild(DOCS[:1])
    assert len(index.search("payment card", 5)) == 1


def test_retriever_filters_low_scores_and_adds_score_metadata():
    index = MemoryIndex(HashingEmbeddings())
    index.rebuild(DOCS)
    docs = KnowledgeRetriever(index=index, k=2, min_score=0.2).invoke("vehicle height")
    assert [d.metadata["doc_id"] for d in docs] == ["rules"]
    assert docs[0].metadata["score"] >= 0.2
