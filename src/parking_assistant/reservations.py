"""Reservation domain model and transactional service."""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from parking_assistant.availability import free_spaces, peak_occupancy
from parking_assistant.car_number import is_valid_car_number, normalize_car_number
from parking_assistant.db import Database
from parking_assistant.timeutils import from_utc_text, to_utc_text, utc_now

MAX_NAME_LENGTH = 50
MAX_CAR_DISPLAY_LENGTH = 15
MIN_DURATION = timedelta(minutes=30)
MAX_DURATION = timedelta(days=30)
MAX_ADVANCE = timedelta(days=365)
RESERVATION_ID = re.compile(r"^[0-9a-f]{32}$")
_NAME = re.compile(r"^[^\W\d_]+(?:[ '\-][^\W\d_]+)*$")


class Status(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ExportStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    RECORDED = "recorded"
    FAILED = "failed"


class ReservationError(Exception):
    """Expected business-rule failure with a user-safe message."""


class NotFoundError(ReservationError):
    pass


class ConflictError(ReservationError):
    pass


def validate_name(value: str, label: str = "Name") -> str:
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError(f"{label} is required.")
    if len(cleaned) > MAX_NAME_LENGTH:
        raise ValueError(f"{label} must be at most {MAX_NAME_LENGTH} characters.")
    if not _NAME.match(cleaned):
        raise ValueError(
            f"{label} may contain only letters, spaces, hyphens and apostrophes."
        )
    return cleaned


def validate_car_number(value: str) -> tuple[str, str]:
    """Return ``(display, normalized)`` or raise ``ValueError``."""
    display = " ".join(value.split()).upper()
    if "|" in display or len(display) > MAX_CAR_DISPLAY_LENGTH:
        raise ValueError("Car number is not valid.")
    normalized = normalize_car_number(display)
    if not is_valid_car_number(normalized):
        raise ValueError(
            "Car number must contain 4-10 Latin letters or digits "
            "(spaces and hyphens are ignored)."
        )
    return display, normalized


def validate_period(start: datetime, end: datetime, now: datetime) -> None:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Datetimes must include a timezone.")
    if end <= start:
        raise ValueError("The end must be after the start.")
    if start <= now:
        raise ValueError("The start must be in the future.")
    if end - start < MIN_DURATION:
        raise ValueError("The minimum reservation is 30 minutes.")
    if end - start > MAX_DURATION:
        raise ValueError("The maximum reservation is 30 days.")
    if start - now > MAX_ADVANCE:
        raise ValueError("Reservations can be made at most 365 days in advance.")


class ReservationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    first_name: str
    last_name: str
    car_number: str
    start: datetime
    end: datetime

    @field_validator("first_name")
    @classmethod
    def _first(cls, value: str) -> str:
        return validate_name(value, "First name")

    @field_validator("last_name")
    @classmethod
    def _last(cls, value: str) -> str:
        return validate_name(value, "Last name")

    @field_validator("car_number")
    @classmethod
    def _car(cls, value: str) -> str:
        return validate_car_number(value)[0]

    @model_validator(mode="after")
    def _period(self) -> ReservationRequest:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("Datetimes must include a timezone.")
        if self.end <= self.start:
            raise ValueError("The end must be after the start.")
        return self

    @property
    def car_number_norm(self) -> str:
        return normalize_car_number(self.car_number)


@dataclass(frozen=True)
class Reservation:
    id: str
    session_id: str
    first_name: str
    last_name: str
    car_number: str
    car_number_norm: str
    start: datetime
    end: datetime
    status: Status
    export_status: ExportStatus
    export_error: str | None
    created_at: datetime
    queued_at: datetime | None
    decided_at: datetime | None
    decided_by: str | None
    decision_reason: str | None
    recorded_at: datetime | None
    notification: str | None = None

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Reservation:
        def opt(key: str) -> datetime | None:
            return from_utc_text(row[key]) if row[key] else None

        return cls(
            id=row["id"],
            session_id=row["session_id"],
            first_name=row["first_name"],
            last_name=row["last_name"],
            car_number=row["car_number"],
            car_number_norm=row["car_number_norm"],
            start=from_utc_text(row["start_utc"]),
            end=from_utc_text(row["end_utc"]),
            status=Status(row["status"]),
            export_status=ExportStatus(row["export_status"]),
            export_error=row["export_error"],
            created_at=from_utc_text(row["created_at"]),
            queued_at=opt("queued_at"),
            decided_at=opt("decided_at"),
            decided_by=row["decided_by"],
            decision_reason=row["decision_reason"],
            recorded_at=opt("recorded_at"),
            notification=row["notification"],
        )


def estimate_price(start: datetime, end: datetime, hourly_rate: Decimal) -> Decimal:
    """Each started hour is charged in full."""
    hours = math.ceil((end - start).total_seconds() / 3600)
    return hourly_rate * hours


class ReservationService:
    def __init__(self, db: Database, clock=utc_now) -> None:
        self.db = db
        self.clock = clock

    # ---- reads -------------------------------------------------------------

    def get(self, reservation_id: str) -> Reservation:
        if not RESERVATION_ID.match(reservation_id or ""):
            raise NotFoundError("Reservation not found.")
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM reservations WHERE id = ?", (reservation_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError("Reservation not found.")
        return Reservation.from_row(row)

    def get_for_session(self, reservation_id: str, session_id: str) -> Reservation:
        """Owner-scoped read: another session's request looks like it does not exist."""
        reservation = self.get(reservation_id)
        if reservation.session_id != session_id:
            raise NotFoundError("Reservation not found.")
        return reservation

    def list_for_session(self, session_id: str) -> list[Reservation]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM reservations WHERE session_id = ? "
                "ORDER BY created_at DESC",
                (session_id,),
            ).fetchall()
        return [Reservation.from_row(r) for r in rows]

    def list_all(self, status: Status | None = None) -> list[Reservation]:
        with self.db.connection() as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM reservations ORDER BY created_at"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM reservations WHERE status = ? ORDER BY created_at",
                    (status.value,),
                ).fetchall()
        return [Reservation.from_row(r) for r in rows]

    def approved_intervals(
        self, conn: sqlite3.Connection, start: datetime, end: datetime
    ) -> list[tuple[datetime, datetime]]:
        rows = conn.execute(
            "SELECT start_utc, end_utc FROM reservations "
            "WHERE status = 'approved' AND start_utc < ? AND end_utc > ?",
            (to_utc_text(end), to_utc_text(start)),
        ).fetchall()
        return [(from_utc_text(r[0]), from_utc_text(r[1])) for r in rows]

    def free_spaces(self, start: datetime, end: datetime) -> int:
        total = self.db.get_settings().total_spaces
        with self.db.connection() as conn:
            intervals = self.approved_intervals(conn, start, end)
        return free_spaces(total, intervals, (start, end))

    def quote(self, start: datetime, end: datetime) -> Decimal:
        return estimate_price(start, end, self.db.get_settings().hourly_rate_gel)

    # ---- writes ------------------------------------------------------------

    def submit(self, session_id: str, request: ReservationRequest) -> Reservation:
        """Create a pending request. Re-submitting the same data is idempotent."""
        now = self.clock()
        try:
            validate_period(request.start, request.end, now)
        except ValueError as exc:
            raise ReservationError(str(exc)) from exc

        key_source = "|".join(
            [
                session_id,
                request.first_name.lower(),
                request.last_name.lower(),
                request.car_number_norm,
                to_utc_text(request.start),
                to_utc_text(request.end),
            ]
        )
        submission_key = hashlib.sha256(key_source.encode()).hexdigest()

        with self.db.transaction() as conn:
            existing = conn.execute(
                "SELECT * FROM reservations WHERE submission_key = ?",
                (submission_key,),
            ).fetchone()
            if existing is not None:
                found = Reservation.from_row(existing)
                if found.status is Status.REJECTED:
                    raise ConflictError(
                        "An identical request was already rejected. "
                        "Please change the period or details."
                    )
                return found

            if self._car_overlap(conn, request.car_number_norm, request, None, True):
                raise ConflictError(
                    "This car already has an active or pending reservation "
                    "that overlaps the requested period."
                )
            reservation_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO reservations (id, session_id, first_name, last_name, "
                "car_number, car_number_norm, start_utc, end_utc, status, "
                "created_at, submission_key) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                (
                    reservation_id,
                    session_id,
                    request.first_name,
                    request.last_name,
                    request.car_number,
                    request.car_number_norm,
                    to_utc_text(request.start),
                    to_utc_text(request.end),
                    to_utc_text(now),
                    submission_key,
                ),
            )
            row = conn.execute(
                "SELECT * FROM reservations WHERE id = ?", (reservation_id,)
            ).fetchone()
        return Reservation.from_row(row)

    def mark_queued(self, reservation_id: str) -> Reservation:
        self.get(reservation_id)
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE reservations SET queued_at = COALESCE(queued_at, ?) "
                "WHERE id = ?",
                (to_utc_text(self.clock()), reservation_id),
            )
        return self.get(reservation_id)

    def approve(self, reservation_id: str, admin: str) -> Reservation:
        """Human approval: re-check rules and allocate capacity atomically.

        Repeating an approval returns the already-approved reservation.
        """
        self.get(reservation_id)
        now = self.clock()
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM reservations WHERE id = ?", (reservation_id,)
            ).fetchone()
            current = Reservation.from_row(row)
            if current.status is Status.APPROVED:
                return current
            if current.status is Status.REJECTED:
                raise ConflictError("The request was already rejected.")
            if current.start <= now:
                raise ConflictError("The reservation start is no longer in the future.")

            total = self._total_spaces(conn)
            intervals = self.approved_intervals(conn, current.start, current.end)
            if peak_occupancy(intervals, (current.start, current.end)) >= total:
                raise ConflictError("No free space remains for this period.")
            if self._car_overlap(
                conn, current.car_number_norm, current, current.id, False
            ):
                raise ConflictError(
                    "This car already has an approved overlapping reservation."
                )

            conn.execute(
                "UPDATE reservations SET status = 'approved', decided_at = ?, "
                "decided_by = ? "
                "WHERE id = ? AND status = 'pending'",
                (to_utc_text(now), admin, reservation_id),
            )
        return self.get(reservation_id)

    def reject(self, reservation_id: str, admin: str, reason: str = "") -> Reservation:
        self.get(reservation_id)
        now = self.clock()
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT status FROM reservations WHERE id = ?", (reservation_id,)
            ).fetchone()
            status = Status(row["status"])
            if status is Status.APPROVED:
                raise ConflictError("The request was already approved.")
            if status is Status.PENDING:
                conn.execute(
                    "UPDATE reservations SET status = 'rejected', decided_at = ?, "
                    "decided_by = ?, decision_reason = ? WHERE id = ?",
                    (to_utc_text(now), admin, reason[:200], reservation_id),
                )
        return self.get(reservation_id)

    def set_notification(self, reservation_id: str, text: str) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE reservations SET notification = ? WHERE id = ?",
                (text[:500], reservation_id),
            )

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _total_spaces(conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT value FROM facility_settings WHERE key = 'total_spaces'"
        ).fetchone()
        return int(row[0])

    @staticmethod
    def _car_overlap(
        conn: sqlite3.Connection,
        car_norm: str,
        period: ReservationRequest | Reservation,
        exclude_id: str | None,
        include_pending: bool,
    ) -> bool:
        statuses = ("pending", "approved") if include_pending else ("approved",)
        placeholders = ",".join("?" for _ in statuses)
        row = conn.execute(
            f"SELECT 1 FROM reservations WHERE car_number_norm = ? "
            f"AND status IN ({placeholders}) AND start_utc < ? AND end_utc > ? "
            f"AND id != ? LIMIT 1",
            (
                car_norm,
                *statuses,
                to_utc_text(period.end),
                to_utc_text(period.start),
                exclude_id or "",
            ),
        ).fetchone()
        return row is not None


def describe_status(r: Reservation) -> str:
    if r.status is Status.PENDING:
        return "waiting for administrator approval"
    if r.status is Status.REJECTED:
        reason = f" ({r.decision_reason})" if r.decision_reason else ""
        return f"rejected by the administrator{reason}"
    return "approved - your reservation is confirmed"
