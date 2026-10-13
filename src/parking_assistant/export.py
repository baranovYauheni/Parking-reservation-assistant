"""Approved-reservation export file.

Line format: ``Name | Car Number | Reservation Period | Approval Time``.
The file is rebuilt as a full snapshot of approved reservations and swapped in
atomically, so retries never duplicate lines and a crash never leaves a
half-written file.
"""

from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from parking_assistant.reservations import Reservation, Status
from parking_assistant.timeutils import local_iso

FORBIDDEN = ("|", "\n", "\r")


class ExportError(Exception):
    pass


def format_line(reservation: Reservation) -> str:
    if reservation.status is not Status.APPROVED or reservation.decided_at is None:
        raise ExportError("Only approved reservations can be exported.")
    fields = [
        reservation.full_name,
        reservation.car_number,
        f"{local_iso(reservation.start)}/{local_iso(reservation.end)}",
        local_iso(reservation.decided_at),
    ]
    for value in fields[:2]:
        if any(ch in value for ch in FORBIDDEN):
            raise ExportError("Field contains a forbidden character.")
    return " | ".join(fields)


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def write_snapshot(path: Path, reservations: Iterable[Reservation]) -> int:
    """Atomically replace ``path`` with one line per approved reservation."""
    ordered = sorted(
        (r for r in reservations if r.status is Status.APPROVED),
        key=lambda r: (r.decided_at, r.id),
    )
    lines = [format_line(r) for r in ordered]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".reservations-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write("".join(line + "\n" for line in lines))
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return len(lines)
