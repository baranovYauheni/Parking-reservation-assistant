import pytest
from fastapi.testclient import TestClient

from parking_assistant.api import create_app
from parking_assistant.container import build_container
from parking_assistant.embeddings import HashingEmbeddings
from parking_assistant.vectorstore import MemoryIndex


@pytest.fixture
def client(settings, clock):
    def factory(s):
        return build_container(
            s,
            clock=clock,
            index=MemoryIndex(HashingEmbeddings()),
            use_configured_llm=False,
        )

    with TestClient(create_app(settings, factory)) as c:
        yield c


def book(client):
    for text in [
        "book",
        "name: John Smith, car: AB123CD",
        "from: 2026-10-20 10:00, to: 2026-10-20 12:00",
    ]:
        client.post("/api/chat", json={"message": text})
    data = client.post("/api/chat", json={"message": "confirm"}).json()
    assert data["route"] == "submitted", data
    return client.get("/api/my-requests").json()["requests"][0]["id"]


def test_chat_answers_with_sources_and_sets_session_cookie(client):
    response = client.post(
        "/api/chat", json={"message": "What is the maximum vehicle height?"}
    )
    assert response.status_code == 200
    assert "2.1 metres" in response.json()["reply"]
    assert response.json()["sources"]
    assert "parking_session" in response.cookies or client.cookies.get(
        "parking_session"
    )


def test_other_session_cannot_see_request(client):
    rid = book(client)
    other = TestClient(client.app)
    assert other.get(f"/api/my-requests/{rid}").status_code == 404
    assert other.get("/api/my-requests").json()["requests"] == []


def test_availability_endpoint_validates_period(client):
    ok = client.get(
        "/api/availability",
        params={"start": "2026-10-20 10:00", "end": "2026-10-20 12:00"},
    )
    assert ok.json()["free_spaces"] == 10
    assert ok.json()["estimated_price_gel"] == "10"
    bad = client.get(
        "/api/availability",
        params={"start": "2026-10-20 12:00", "end": "2026-10-20 10:00"},
    )
    assert bad.status_code == 422
