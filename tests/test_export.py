from dataclasses import replace
from datetime import UTC, datetime

import pytest

from parking_assistant.export import ExportError, format_line, write_snapshot
from parking_assistant.reservations import ExportStatus, Reservation, Status

T = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)


def reservation(rid: str = "a" * 32, **changes) -> Reservation:
    base = Reservation(
        id=rid,
        session_id="s",
        first_name="John",
        last_name="Smith",
        car_number="AB 123 CD",
        car_number_norm="AB123CD",
        start=T,
        end=T.replace(hour=12),
        status=Status.APPROVED,
        export_status=ExportStatus.PENDING,
        export_error=None,
        created_at=T.replace(hour=6),
        queued_at=None,
        decided_at=T.replace(hour=7),
        decided_by="admin",
        decision_reason=None,
        recorded_at=None,
    )
    return replace(base, **changes)


def test_line_format_matches_specification():
    assert format_line(reservation()) == (
        "John Smith | AB 123 CD | 2026-10-20T14:00:00+04:00/2026-10-20T16:00:00+04:00"
        " | 2026-10-20T11:00:00+04:00"
    )


def test_only_approved_reservations_can_be_formatted():
    with pytest.raises(ExportError):
        format_line(reservation(status=Status.PENDING, decided_at=None))
    with pytest.raises(ExportError):
        format_line(reservation(last_name="Smith | X"))


def test_snapshot_rewrite_is_idempotent_and_skips_unapproved(tmp_path):
    path = tmp_path / "out" / "reservations.txt"
    items = [reservation("a" * 32), reservation("b" * 32, status=Status.REJECTED)]
    assert write_snapshot(path, items) == 1
    assert write_snapshot(path, items) == 1
    assert len(path.read_text().splitlines()) == 1
    assert not list(path.parent.glob(".reservations-*"))
