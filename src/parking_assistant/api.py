"""FastAPI application: visitor chat, admin approval API and HTML pages."""

import hmac
import logging
import re
import secrets
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from parking_assistant.config import Settings
from parking_assistant.container import Container, build_container
from parking_assistant.reservations import (
    ConflictError,
    NotFoundError,
    Reservation,
    ReservationError,
    Status,
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


class DecisionIn(BaseModel):
    action: Literal["approve", "reject"]
    reason: str = Field(default="", max_length=200)


def public_view(r: Reservation) -> dict:
    """What the owner of a request may see."""
    return {
        "id": r.id,
        "name": r.full_name,
        "car_number": r.car_number,
        "start": local_iso(r.start),
        "end": local_iso(r.end),
        "status": r.status.value,
        "export_status": r.export_status.value,
        "status_text": describe_status(r),
        "notification": r.notification,
    }


def admin_view(r: Reservation, free_spaces: int | None = None) -> dict:
    data = public_view(r) | {
        "created_at": local_iso(r.created_at),
        "queued_at": local_iso(r.queued_at) if r.queued_at else None,
        "decided_at": local_iso(r.decided_at) if r.decided_at else None,
        "decided_by": r.decided_by,
        "decision_reason": r.decision_reason,
        "export_error": r.export_error,
        "recorded_at": local_iso(r.recorded_at) if r.recorded_at else None,
    }
    if free_spaces is not None:
        data["free_spaces_for_period"] = free_spaces
    return data


def create_app(
    settings: Settings | None = None,
    container_factory: Callable[[Settings], Container] | None = None,
) -> FastAPI:
    settings = settings or Settings()
    factory = container_factory or build_container

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        token_was_configured = settings.admin_token is not None
        container = factory(settings)
        app.state.container = container
        if not token_was_configured:
            logger.warning(
                "ADMIN_TOKEN not set; generated admin token for this run: %s",
                settings.admin_token.get_secret_value(),
            )
        resumed = container.workflow.recover()
        if resumed:
            logger.info("resumed %d workflow(s) after restart", len(resumed))
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

    def require_admin(
        authorization: Annotated[str | None, Header()] = None,
    ) -> str:
        expected = (
            settings.admin_token.get_secret_value() if settings.admin_token else ""
        )
        scheme, _, token = (authorization or "").partition(" ")
        if (
            not expected
            or scheme.lower() != "bearer"
            or not hmac.compare_digest(token.encode(), expected.encode())
        ):
            raise HTTPException(
                status_code=401,
                detail="Administrator authentication required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return "admin"

    ContainerDep = Annotated[Container, Depends(get_container)]
    SessionDep = Annotated[str, Depends(get_session)]
    AdminDep = Annotated[str, Depends(require_admin)]

    # ---- pages -----------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def chat_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "chat.html")

    @app.get("/admin", include_in_schema=False)
    def admin_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "admin.html")

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

    # ---- admin API -------------------------------------------------------------

    @app.get("/api/admin/requests")
    def admin_list(c: ContainerDep, _: AdminDep, status: Status | None = None) -> dict:
        items = []
        for r in c.service.list_all(status):
            free = (
                c.service.free_spaces(r.start, r.end)
                if r.status is Status.PENDING
                else None
            )
            items.append(admin_view(r, free))
        return {"requests": items}

    @app.get("/api/admin/requests/{reservation_id}")
    def admin_get(reservation_id: str, c: ContainerDep, _: AdminDep) -> dict:
        try:
            r = c.service.get(reservation_id)
        except NotFoundError as exc:
            raise HTTPException(404, "Request not found.") from exc
        return admin_view(r, c.service.free_spaces(r.start, r.end)) | {
            "workflow_waiting_for": c.workflow.waiting_for(reservation_id)
        }

    @app.post("/api/admin/requests/{reservation_id}/decision")
    def admin_decide(
        reservation_id: str, body: DecisionIn, c: ContainerDep, admin: AdminDep
    ) -> dict:
        try:
            outcome = c.workflow.decide(reservation_id, body.action, admin, body.reason)
        except NotFoundError as exc:
            raise HTTPException(404, "Request not found.") from exc
        except ConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        if outcome.error:
            raise HTTPException(409, outcome.error)
        return admin_view(outcome.reservation)

    @app.post("/api/admin/requests/{reservation_id}/retry-export")
    def admin_retry(reservation_id: str, c: ContainerDep, _: AdminDep) -> dict:
        try:
            r = c.workflow.retry_export(reservation_id)
        except NotFoundError as exc:
            raise HTTPException(404, "Request not found.") from exc
        except ReservationError as exc:
            raise HTTPException(409, str(exc)) from exc
        return admin_view(r)

    @app.get("/api/admin/export")
    def admin_export(c: ContainerDep, _: AdminDep) -> dict:
        path = c.settings.export_path
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        return {
            "format": "Name | Car Number | Reservation Period | Approval Time",
            "lines": lines,
        }

    return app
