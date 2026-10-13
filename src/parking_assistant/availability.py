"""Capacity calculations over half-open intervals ``[start, end)``."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

Interval = tuple[datetime, datetime]


def overlaps(a: Interval, b: Interval) -> bool:
    """True when two half-open intervals share at least one instant."""
    return a[0] < b[1] and b[0] < a[1]


def peak_occupancy(bookings: Iterable[Interval], window: Interval) -> int:
    """Maximum number of bookings active at the same instant inside ``window``.

    Sweep line: each booking contributes +1 at its (clipped) start and -1 at its
    end. Ends are processed before starts at the same instant, so back-to-back
    bookings do not count as simultaneous. O(n log n).
    """
    win_start, win_end = window
    events: list[tuple[datetime, int]] = []
    for start, end in bookings:
        if not overlaps((start, end), window):
            continue
        events.append((max(start, win_start), 1))
        events.append((min(end, win_end), -1))
    events.sort(key=lambda event: (event[0], event[1]))

    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def free_spaces(total: int, bookings: Iterable[Interval], window: Interval) -> int:
    return max(total - peak_occupancy(bookings, window), 0)
