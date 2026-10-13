from langchain_core.language_models.fake_chat_models import FakeListChatModel

from parking_assistant.embeddings import HashingEmbeddings
from parking_assistant.knowledge import load_chunks
from parking_assistant.rag import RagAnswerer, detect_topics
from parking_assistant.vectorstore import KnowledgeRetriever, MemoryIndex
from tests.conftest import KNOWLEDGE_DIR


def make_answerer(db, service, llm=None):
    index = MemoryIndex(HashingEmbeddings())
    index.rebuild(load_chunks(KNOWLEDGE_DIR))
    retriever = KnowledgeRetriever(index=index, k=3, min_score=0.1)
    return RagAnswerer(retriever, db, service, llm)


def test_document_question_is_answered_with_source(db, service):
    answer = make_answerer(db, service).answer("What is the maximum vehicle height?")
    assert "2.1 metres" in answer.text
    assert answer.sources[0]["doc_id"] == "rules"


def test_prices_come_from_sql_not_documents(db, service):
    answerer = make_answerer(db, service)
    assert "GEL 5.00" in answerer.answer("How much does parking cost?").text
    db.set_setting("hourly_rate_gel", "7.50")
    answer = answerer.answer("How much does parking cost?")
    assert "GEL 7.50" in answer.text
    assert answer.sources[0]["doc_id"] == "live-data"


def test_availability_for_period_in_question(db, service):
    answer = make_answerer(db, service).answer(
        "Are there free spaces from 2026-10-20 10:00 to 2026-10-20 12:00?"
    )
    assert "there are 10 of 10 spaces free" in answer.text


def test_unknown_question_is_answered_honestly(db, service):
    answer = make_answerer(db, service).answer("Do you sell coffee?")
    assert answer.unknown
    assert "don't have information" in answer.text


def test_llm_path_builds_context_and_redacts_output(db, service):
    llm = FakeListChatModel(
        responses=["The height limit is 2.1 metres [rules]. Call +995 555 123 456."]
    )
    answer = make_answerer(db, service, llm).answer("What is the height limit?")
    assert answer.used_llm
    assert "[redacted phone]" in answer.text
    assert [s["doc_id"] for s in answer.sources] == ["rules"]


def test_llm_failure_falls_back_to_offline_answer(db, service):
    class Broken(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise TimeoutError("LLM timeout")

    answer = make_answerer(db, service, Broken(responses=["x"])).answer(
        "What is the maximum vehicle height?"
    )
    assert not answer.used_llm
    assert "2.1 metres" in answer.text


def test_detect_topics():
    assert detect_topics("What are your opening hours and prices?") == {
        "hours",
        "price",
    }
    assert detect_topics("Where is it?") == set()
