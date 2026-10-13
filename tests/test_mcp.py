"""Real MCP client <-> stdio server round trips."""

import sys

from pydantic import SecretStr

from parking_assistant.mcp_client import ExportRecorder
from parking_assistant.reservations import ExportStatus
from tests.conftest import make_request


def test_unapproved_reservation_is_refused_and_nothing_written(
    settings, service, clock
):
    settings.ensure_secrets()
    r = service.submit("s", make_request(clock))
    result = ExportRecorder(settings).record(r.id)
    assert not result.ok and "not approved" in result.detail
    assert not settings.export_path.exists()


def test_approved_reservation_is_recorded_once_even_when_repeated(
    settings, service, clock
):
    settings.ensure_secrets()
    r = service.submit("s", make_request(clock))
    service.approve(r.id, "admin")
    recorder = ExportRecorder(settings)
    assert recorder.record(r.id).ok
    assert recorder.record(r.id).ok
    lines = settings.export_path.read_text().splitlines()
    assert len(lines) == 1 and lines[0].startswith("John Smith | AB123CD | ")
    assert service.get(r.id).export_status is ExportStatus.RECORDED


def test_wrong_token_is_unauthorized(settings, service, clock):
    settings.ensure_secrets()
    r = service.submit("s", make_request(clock))
    service.approve(r.id, "admin")
    attacker = settings.model_copy(update={"mcp_shared_secret": SecretStr("guess")})
    result = ExportRecorder(attacker).record(r.id)
    assert not result.ok and "unauthorized" in result.detail
    assert not settings.export_path.exists()


def test_unavailable_server_reports_failure_instead_of_success(
    settings, service, clock
):
    settings.ensure_secrets()
    r = service.submit("s", make_request(clock))
    service.approve(r.id, "admin")
    broken = ExportRecorder(
        settings, command=[sys.executable, "-c", "raise SystemExit(1)"]
    )
    result = broken.record(r.id)
    assert not result.ok and "unavailable" in result.detail
