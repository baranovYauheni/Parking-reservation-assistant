"""SQLite storage: schema, connections and dynamic facility settings."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from parking_assistant import facility

SCHEMA = """
CREATE TABLE IF NOT EXISTS facility_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reservations (
    id               TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    first_name       TEXT NOT NULL,
    last_name        TEXT NOT NULL,
    car_number       TEXT NOT NULL,
    car_number_norm  TEXT NOT NULL,
    start_utc        TEXT NOT NULL,
    end_utc          TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'approved', 'rejected')),
    export_status    TEXT NOT NULL DEFAULT 'not_required'
                     CHECK (export_status IN
                            ('not_required', 'pending', 'recorded', 'failed')),
    export_error     TEXT,
    created_at       TEXT NOT NULL,
    queued_at        TEXT,
    decided_at       TEXT,
    decided_by       TEXT,
    decision_reason  TEXT,
    recorded_at      TEXT,
    notification     TEXT,
    submission_key   TEXT NOT NULL UNIQUE,
    CHECK (end_utc > start_utc)
);

CREATE INDEX IF NOT EXISTS ix_res_status_period
    ON reservations (status, start_utc, end_utc);
CREATE INDEX IF NOT EXISTS ix_res_session ON reservations (session_id);
CREATE INDEX IF NOT EXISTS ix_res_car ON reservations (car_number_norm, status);
"""

DEFAULT_SETTINGS: dict[str, str] = {
    "facility_name": facility.FACILITY_NAME,
    "address": "12 Example Avenue, Demo District, Tbilisi (fictional)",
    "total_spaces": str(facility.TOTAL_SPACES),
    "hourly_rate_gel": str(facility.HOURLY_RATE_GEL),
    "opening_hours": facility.OPENING_HOURS,
    "timezone": facility.TIMEZONE_NAME,
}


@dataclass(frozen=True)
class FacilitySettings:
    facility_name: str
    address: str
    total_spaces: int
    hourly_rate_gel: Decimal
    opening_hours: str
    timezone: str


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 15000")
    return conn


class Database:
    """Opens a short-lived connection per unit of work (safe across threads)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)
            for key, value in DEFAULT_SETTINGS.items():
                conn.execute(
                    "INSERT OR IGNORE INTO facility_settings (key, value) "
                    "VALUES (?, ?)",
                    (key, value),
                )

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = connect(self.path)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Write transaction that takes the database write lock up front."""
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def get_settings(self) -> FacilitySettings:
        with self.connection() as conn:
            rows = conn.execute("SELECT key, value FROM facility_settings").fetchall()
        values = DEFAULT_SETTINGS | {row["key"]: row["value"] for row in rows}
        return FacilitySettings(
            facility_name=values["facility_name"],
            address=values["address"],
            total_spaces=int(values["total_spaces"]),
            hourly_rate_gel=Decimal(values["hourly_rate_gel"]),
            opening_hours=values["opening_hours"],
            timezone=values["timezone"],
        )

    def set_setting(self, key: str, value: str) -> None:
        if key not in DEFAULT_SETTINGS:
            raise KeyError(key)
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO facility_settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
