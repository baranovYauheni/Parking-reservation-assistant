import pytest
from fastapi.testclient import TestClient

from parking_assistant.api import create_app
from parking_assistant.container import build_container
from parking_assistant.embeddings import HashingEmbeddings
from parking_assistant.vectorstore import MemoryIndex

ADMIN = {"Authorization": "Bearer test-admin-token"}


@pytest.fixture
def client(settings, clock, recorder):
    def factory(s):
        return build_container(
            s,
            clock=clock,
            index=MemoryIndex(HashingEmbeddings()),
            recorder=recorder,
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


def test_admin_endpoints_require_valid_token(client):
    assert client.get("/api/admin/requests").status_code == 401
    bad = {"Authorization": "Bearer nope"}
    assert client.get("/api/admin/requests", headers=bad).status_code == 401
    assert client.get("/api/admin/requests", headers=ADMIN).status_code == 200


def test_other_session_cannot_see_request(client, settings, clock, recorder):
    rid = book(client)
    other = TestClient(client.app)
    assert other.get(f"/api/my-requests/{rid}").status_code == 404
    assert other.get("/api/my-requests").json()["requests"] == []


def test_approve_flow_writes_export_and_updates_user_status(client):
    rid = book(client)
    pending = client.get("/api/admin/requests?status=pending", headers=ADMIN).json()[
        "requests"
    ]
    assert [r["id"] for r in pending] == [rid]
    response = client.post(
        f"/api/admin/requests/{rid}/decision", json={"action": "approve"}, headers=ADMIN
    )
    assert response.status_code == 200
    assert response.json()["export_status"] == "recorded"
    export = client.get("/api/admin/export", headers=ADMIN).json()["lines"]
    assert export == [
        "John Smith | AB123CD | 2026-10-20T10:00:00+04:00/2026-10-20T12:00:00+04:00 | 2026-10-13T12:00:00+04:00"
    ]
    mine = client.get(f"/api/my-requests/{rid}").json()
    assert mine["status_text"].startswith("approved and recorded")


def test_reject_then_approve_conflicts(client):
    rid = book(client)
    assert (
        client.post(
            f"/api/admin/requests/{rid}/decision",
            json={"action": "reject"},
            headers=ADMIN,
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"/api/admin/requests/{rid}/decision",
            json={"action": "approve"},
            headers=ADMIN,
        ).status_code
        == 409
    )
    assert client.get("/api/admin/export", headers=ADMIN).json()["lines"] == []


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


def test_unknown_request_id_returns_404(client):
    assert (
        client.post(
            "/api/admin/requests/" + "0" * 32 + "/decision",
            json={"action": "approve"},
            headers=ADMIN,
        ).status_code
        == 404
    )
    assert client.get("/api/admin/requests/not-an-id", headers=ADMIN).status_code == 404
