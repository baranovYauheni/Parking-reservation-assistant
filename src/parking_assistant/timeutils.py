"""Datetime parsing and formatting in the parking timezone.

All stored values are timezone-aware UTC datetimes. User input without an
explicit offset is interpreted in the parking timezone.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from parking_assistant.facility import TIMEZONE_NAME

PARKING_TZ = ZoneInfo(TIMEZONE_NAME)

Clock = Callable[[], datetime]

_RELATIVE = re.compile(
    r"^(today|tomorrow)\s+(?:at\s+)?(\d{1,2}):(\d{2})$", re.IGNORECASE
)
_LOCAL_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%d.%m.%Y %H:%M")


class DateParseError(ValueError):
    pass


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_user_datetime(text: str, now: datetime | None = None) -> datetime:
    """Parse a user-supplied datetime and return it in UTC.

    Accepted: ISO 8601 with offset (``2026-10-20T10:00+04:00``), local
    ``YYYY-MM-DD HH:MM``, ``DD.MM.YYYY HH:MM`` and ``today/tomorrow HH:MM``.
    A date without a time is ambiguous and rejected.
    """
    value = text.strip()
    if not value:
        raise DateParseError("Please enter a date and time.")
    now = now or utc_now()

    match = _RELATIVE.match(value)
    if match:
        word, hour, minute = match.groups()
        local_today = now.astimezone(PARKING_TZ).date()
        day = local_today + timedelta(days=1 if word.lower() == "tomorrow" else 0)
        try:
            local = datetime(
                day.year, day.month, day.day, int(hour), int(minute), tzinfo=PARKING_TZ
            )
        except ValueError as exc:
            raise DateParseError(f"Invalid time: {value!r}.") from exc
        return local.astimezone(UTC)

    iso = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        parsed = None
    if parsed is not None:
        if len(iso) <= 10:
            raise DateParseError("Please include a time, for example 2026-10-20 14:00.")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=PARKING_TZ)
        return parsed.astimezone(UTC)

    for fmt in _LOCAL_FORMATS:
        try:
            local = datetime.strptime(value, fmt).replace(tzinfo=PARKING_TZ)
        except ValueError:
            continue
        return local.astimezone(UTC)

    raise DateParseError(
        f"I could not understand {value!r}. Use the format YYYY-MM-DD HH:MM "
        f"({TIMEZONE_NAME} time), for example 2026-10-20 14:00."
    )


def to_utc_text(value: datetime) -> str:
    """Serialize a datetime as a sortable UTC ISO string."""
    if value.tzinfo is None:
        raise ValueError("naive datetime")
    return value.astimezone(UTC).isoformat(timespec="seconds")


def from_utc_text(text: str) -> datetime:
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        raise ValueError("stored datetime without offset")
    return value.astimezone(UTC)


def to_local(value: datetime) -> datetime:
    return value.astimezone(PARKING_TZ)


def local_iso(value: datetime) -> str:
    """ISO 8601 in parking time with offset, e.g. 2026-10-20T10:00:00+04:00."""
    return to_local(value).isoformat(timespec="seconds")


def human_local(value: datetime) -> str:
    return to_local(value).strftime("%Y-%m-%d %H:%M") + f" ({TIMEZONE_NAME})"
