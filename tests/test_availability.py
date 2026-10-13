from datetime import UTC, datetime, timedelta

from parking_assistant.availability import free_spaces, overlaps, peak_occupancy

T0 = datetime(2026, 10, 20, 8, 0, tzinfo=UTC)


def h(n: float) -> datetime:
    return T0 + timedelta(hours=n)


def test_half_open_intervals_back_to_back_do_not_overlap():
    assert not overlaps((h(0), h(2)), (h(2), h(4)))
    assert overlaps((h(0), h(2)), (h(1), h(3)))
    assert peak_occupancy([(h(0), h(2)), (h(2), h(4))], (h(0), h(4))) == 1


def test_peak_differs_from_number_of_overlapping_bookings():
    # Three bookings overlap the window, but never more than two at once.
    bookings = [(h(0), h(2)), (h(1), h(3)), (h(2), h(4))]
    assert peak_occupancy(bookings, (h(0), h(4))) == 2
    assert free_spaces(2, bookings, (h(0), h(4))) == 0
    assert free_spaces(3, bookings, (h(0), h(4))) == 1


def test_bookings_outside_window_are_ignored_and_clipped():
    bookings = [(h(-5), h(-1)), (h(-1), h(1)), (h(10), h(12))]
    assert peak_occupancy(bookings, (h(0), h(2))) == 1
    assert peak_occupancy([], (h(0), h(2))) == 0
