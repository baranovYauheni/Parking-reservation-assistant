import threading
from datetime import timedelta

import pytest
from pydantic import ValidationError

from parking_assistant.reservations import (
    ConflictError,
    NotFoundError,
    ReservationError,
    ReservationRequest,
    ReservationService,
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


def test_pending_requests_do_not_consume_capacity(service, db, clock):
    db.set_setting("total_spaces", "1")
    a = service.submit("s1", make_request(clock, car="CAR0001"))
    b = service.submit("s2", make_request(clock, car="CAR0002"))
    start, end = a.start, a.end
    assert service.free_spaces(start, end) == 1
    service.approve(a.id, "admin")
    assert service.free_spaces(start, end) == 0
    with pytest.raises(ConflictError, match="No free space"):
        service.approve(b.id, "admin")
    assert service.get(b.id).status is Status.PENDING


def test_repeated_approval_is_idempotent_and_conflicting_decision_fails(service, clock):
    r = service.submit("s", make_request(clock))
    first = service.approve(r.id, "admin")
    again = service.approve(r.id, "admin")
    assert first.decided_at == again.decided_at
    with pytest.raises(ConflictError):
        service.reject(r.id, "admin")


def test_approval_rechecks_that_start_is_still_in_future(service, clock):
    r = service.submit("s", make_request(clock, start_h=1))
    clock.advance(hours=2)
    with pytest.raises(ConflictError, match="future"):
        service.approve(r.id, "admin")


def test_concurrent_approvals_never_overbook(db, clock):
    db.set_setting("total_spaces", "3")
    service = ReservationService(db, clock=clock)
    ids = [
        service.submit(f"s{i}", make_request(clock, car=f"CAR{i:04d}")).id
        for i in range(8)
    ]
    results: list[str] = []

    def approve(rid: str) -> None:
        try:
            service.approve(rid, "admin")
            results.append("ok")
        except ConflictError:
            results.append("conflict")

    threads = [threading.Thread(target=approve, args=(rid,)) for rid in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("ok") == 3
    assert len(service.list_all(Status.APPROVED)) == 3


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
