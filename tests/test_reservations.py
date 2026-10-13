from datetime import timedelta

import pytest
from pydantic import ValidationError

from parking_assistant.reservations import (
    ConflictError,
    NotFoundError,
    ReservationError,
    ReservationRequest,
    Status,
    estimate_price,
)
from tests.conftest import make_request


def test_submit_creates_pending_request_and_is_idempotent(service, clock):
    request = make_request(clock)
    first = service.submit("session-a", request)
    second = service.submit("session-a", request)
    assert first.status is Status.PENDING
    assert first.id == second.id
    assert len(service.list_all()) == 1


@pytest.mark.parametrize(
    ("start_h", "hours", "message"),
    [(-1, 2, "future"), (1, 0.25, "minimum"), (1, 24 * 31, "maximum")],
)
def test_invalid_periods_are_rejected(service, clock, start_h, hours, message):
    with pytest.raises(ReservationError, match=message):
        service.submit("s", make_request(clock, start_h=start_h, hours=hours))


def test_request_model_rejects_bad_fields(clock):
    with pytest.raises(ValidationError):
        make_request(clock, first="J0hn")
    with pytest.raises(ValidationError):
        make_request(clock, car="AB|12")
    with pytest.raises(ValidationError):
        ReservationRequest(
            first_name="A",
            last_name="B",
            car_number="AB123CD",
            start=clock.now + timedelta(hours=2),
            end=clock.now + timedelta(hours=1),
        )


def test_same_car_cannot_have_overlapping_active_requests(service, clock):
    service.submit("s1", make_request(clock, car="AB-123-CD", start_h=24, hours=3))
    with pytest.raises(ConflictError):
        service.submit("s2", make_request(clock, car="ab123cd", start_h=25, hours=3))
    # Back-to-back is fine because intervals are half-open.
    service.submit("s2", make_request(clock, car="ab123cd", start_h=27, hours=1))


def test_other_session_cannot_read_request(service, clock):
    r = service.submit("owner", make_request(clock))
    assert service.get_for_session(r.id, "owner").id == r.id
    with pytest.raises(NotFoundError):
        service.get_for_session(r.id, "intruder")
    assert service.list_for_session("intruder") == []


def test_price_counts_each_started_hour(clock):
    start = clock.now
    from decimal import Decimal

    assert (
        estimate_price(start, start + timedelta(hours=2, minutes=10), Decimal(5)) == 15
    )
    assert estimate_price(start, start + timedelta(hours=1), Decimal(5)) == 5
