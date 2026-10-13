from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from parking_assistant.config import Settings
from parking_assistant.container import build_container
from parking_assistant.db import Database
from parking_assistant.embeddings import HashingEmbeddings
from parking_assistant.mcp_client import RecordResult
from parking_assistant.mcp_server import record_reservation_core
from parking_assistant.reservations import ReservationRequest, ReservationService
from parking_assistant.vectorstore import MemoryIndex

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge"


class FakeClock:
    def __init__(self, now: datetime | None = None) -> None:
        self.now = now or datetime(2026, 10, 13, 8, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


class InProcessRecorder:
    """Calls the MCP tool implementation directly (no subprocess)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.calls: list[str] = []
        self.fail = False

    def record(self, reservation_id: str) -> RecordResult:
        self.calls.append(reservation_id)
        if self.fail:
            return RecordResult(False, "MCP server unavailable: simulated")
        try:
            record_reservation_core(self.settings, reservation_id)
        except Exception as exc:
            return RecordResult(False, str(exc))
        return RecordResult(True, "recorded")


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        PARKING_DATA_DIR=tmp_path / "var",
        PARKING_KNOWLEDGE_DIR=KNOWLEDGE_DIR,
        VECTOR_BACKEND="memory",
        EMBEDDINGS_PROVIDER="hashing",
        LLM_ENABLED=False,
        ADMIN_TOKEN="test-admin-token",
        _env_file=None,
    )


@pytest.fixture
def db(settings: Settings) -> Database:
    settings.prepare_dirs()
    database = Database(settings.db_path)
    database.init()
    return database


@pytest.fixture
def service(db: Database, clock: FakeClock) -> ReservationService:
    return ReservationService(db, clock=clock)


def make_request(
    clock: FakeClock,
    *,
    first: str = "John",
    last: str = "Smith",
    car: str = "AB123CD",
    start_h: float = 24,
    hours: float = 2,
) -> ReservationRequest:
    start = clock.now + timedelta(hours=start_h)
    return ReservationRequest(
        first_name=first,
        last_name=last,
        car_number=car,
        start=start,
        end=start + timedelta(hours=hours),
    )


@pytest.fixture
def recorder(settings: Settings) -> InProcessRecorder:
    return InProcessRecorder(settings)


@pytest.fixture
def container(settings: Settings, clock: FakeClock, recorder: InProcessRecorder):
    built = build_container(
        settings,
        clock=clock,
        index=MemoryIndex(HashingEmbeddings()),
        recorder=recorder,
        use_configured_llm=False,
    )
    yield built
    built.close()
