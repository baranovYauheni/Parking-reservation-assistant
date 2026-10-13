import pytest

from parking_assistant.reservations import ConflictError, Status
from tests.conftest import make_request
from tests.test_chat import submit


def test_submission_is_escalated_by_admin_agent(container):
    rid = submit(container.chat)
    reservation = container.service.get(rid)
    assert reservation.queued_at is not None
    assert container.workflow.waiting_for(rid) == "admin_decision"


def test_human_approval_updates_status_and_notifies_visitor(container):
    rid = submit(container.chat)
    outcome = container.workflow.decide(rid, "approve", "admin")
    assert outcome.error is None
    r = container.service.get(rid)
    assert r.status is Status.APPROVED
    assert "approved" in r.notification
    assert container.workflow.waiting_for(rid) is None


def test_rejection_and_conflicting_second_decision(container):
    rid = submit(container.chat)
    container.workflow.decide(rid, "reject", "admin", "full")
    assert container.service.get(rid).status is Status.REJECTED
    assert container.workflow.decide(rid, "reject", "admin").error is None
    with pytest.raises(ConflictError):
        container.workflow.decide(rid, "approve", "admin")
    assert "rejected" in container.chat.send("sess-1", "status")["reply"]


def test_capacity_conflict_keeps_request_pending(container):
    container.db.set_setting("total_spaces", "1")
    service = container.service
    other = service.submit(
        "other",
        make_request(service.clock, car="ZZ999ZZ", start_h=24 * 7 - 4, hours=24),
    )
    service.approve(other.id, "admin")
    rid = submit(container.chat)
    outcome = container.workflow.decide(rid, "approve", "admin")
    assert outcome.error and "No free space" in outcome.error
    assert service.get(rid).status is Status.PENDING
