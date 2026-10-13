"""FastAPI application: visitor chat API and HTML page."""

import logging
import re
import secrets
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from parking_assistant.config import Settings
from parking_assistant.container import Container, build_container
from parking_assistant.reservations import (
    NotFoundError,
    Reservation,
    describe_status,
)
from parking_assistant.timeutils import (
    DateParseError,
    local_iso,
    parse_user_datetime,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
SESSION_COOKIE = "parking_session"
_SESSION_FORMAT = re.compile(r"^[A-Za-z0-9_-]{32,64}$")


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


def public_view(r: Reservation) -> dict:
    """What the owner of a request may see."""
    return {
        "id": r.id,
        "name": r.full_name,
        "car_number": r.car_number,
        "start": local_iso(r.start),
        "end": local_iso(r.end),
        "status": r.status.value,
        "status_text": describe_status(r),
    }


def create_app(
    settings: Settings | None = None,
    container_factory: Callable[[Settings], Container] | None = None,
) -> FastAPI:
    settings = settings or Settings()
    factory = container_factory or build_container

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = factory(settings)
        app.state.container = container
        yield
        container.close()

    app = FastAPI(title="Parking Reservation Assistant", lifespan=lifespan)

    def get_container(request: Request) -> Container:
        return request.app.state.container

    def get_session(request: Request, response: Response) -> str:
        session_id = request.cookies.get(SESSION_COOKIE, "")
        if not _SESSION_FORMAT.match(session_id):
            session_id = secrets.token_urlsafe(32)
            response.set_cookie(
                SESSION_COOKIE,
                session_id,
                httponly=True,
                samesite="strict",
                max_age=60 * 60 * 24 * 30,
            )
        return session_id

    ContainerDep = Annotated[Container, Depends(get_container)]
    SessionDep = Annotated[str, Depends(get_session)]

    # ---- pages -----------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def chat_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "chat.html")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    # ---- public API ------------------------------------------------------------

    @app.get("/api/facility")
    def facility(c: ContainerDep) -> dict:
        s = c.db.get_settings()
        now = c.service.clock()
        return {
            "name": s.facility_name,
            "address": s.address,
            "total_spaces": s.total_spaces,
            "hourly_rate_gel": str(s.hourly_rate_gel),
            "opening_hours": s.opening_hours,
            "timezone": s.timezone,
            "free_spaces_next_hour": c.service.free_spaces(
                now, now + timedelta(hours=1)
            ),
            "llm": "openai" if c.answerer.llm is not None else "offline",
        }

    @app.get("/api/availability")
    def availability(start: str, end: str, c: ContainerDep) -> dict:
        try:
            start_dt = parse_user_datetime(start)
            end_dt = parse_user_datetime(end)
        except DateParseError as exc:
            raise HTTPException(422, str(exc)) from exc
        if end_dt <= start_dt:
            raise HTTPException(422, "The end must be after the start.")
        return {
            "start": local_iso(start_dt),
            "end": local_iso(end_dt),
            "free_spaces": c.service.free_spaces(start_dt, end_dt),
            "estimated_price_gel": str(c.service.quote(start_dt, end_dt)),
        }

    @app.post("/api/chat")
    def chat(body: ChatIn, c: ContainerDep, session_id: SessionDep) -> dict:
        state = c.chat.send(session_id, body.message)
        booking = state.get("booking") or {}
        return {
            "reply": state["reply"],
            "sources": state.get("sources") or [],
            "route": state.get("route", ""),
            "booking_mode": booking.get("mode", "idle"),
            "awaiting": booking.get("awaiting"),
        }

    @app.get("/api/chat/history")
    def chat_history(c: ContainerDep, session_id: SessionDep) -> dict:
        return {"messages": c.chat.history(session_id)}

    @app.get("/api/my-requests")
    def my_requests(c: ContainerDep, session_id: SessionDep) -> dict:
        return {
            "requests": [public_view(r) for r in c.service.list_for_session(session_id)]
        }

    @app.get("/api/my-requests/{reservation_id}")
    def my_request(
        reservation_id: str, c: ContainerDep, session_id: SessionDep
    ) -> dict:
        try:
            return public_view(c.service.get_for_session(reservation_id, session_id))
        except NotFoundError as exc:
            raise HTTPException(404, "Request not found.") from exc

    return app
