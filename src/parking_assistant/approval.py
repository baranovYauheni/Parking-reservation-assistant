"""Human-in-the-loop approval flow (without orchestration framework).

submit -> admin agent escalates -> human decides via admin API ->
transactional approve/reject -> MCP recording (approved only) ->
admin agent writes the visitor notification.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from parking_assistant.admin_agent import AdminAgent
from parking_assistant.mcp_client import ExportRecorder
from parking_assistant.reservations import (
    ConflictError,
    ExportStatus,
    Reservation,
    ReservationError,
    ReservationService,
    Status,
)

logger = logging.getLogger(__name__)


@dataclass
class DecisionOutcome:
    reservation: Reservation
    error: str | None = None


class ApprovalService:
    def __init__(
        self,
        service: ReservationService,
        admin_agent: AdminAgent,
        recorder: ExportRecorder,
    ) -> None:
        self.service = service
        self.admin_agent = admin_agent
        self.recorder = recorder
        self._locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()

    def _lock(self, reservation_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks[reservation_id]

    def start(self, reservation_id: str) -> None:
        self.admin_agent.escalate(reservation_id)

    def waiting_for(self, reservation_id: str) -> str | None:
        r = self.service.get(reservation_id)
        if r.status is Status.PENDING:
            return "admin_decision"
        if r.status is Status.APPROVED and r.export_status is not ExportStatus.RECORDED:
            return "export_retry"
        return None

    def decide(
        self,
        reservation_id: str,
        action: Literal["approve", "reject"],
        admin: str,
        reason: str = "",
    ) -> DecisionOutcome:
        current = self.service.get(reservation_id)
        with self._lock(reservation_id):
            if current.status is not Status.PENDING:
                same = (action == "approve" and current.status is Status.APPROVED) or (
                    action == "reject" and current.status is Status.REJECTED
                )
                if same:
                    return DecisionOutcome(current)
                raise ConflictError(f"The request is already {current.status.value}.")
            try:
                if action == "approve":
                    self.service.approve(reservation_id, admin)
                else:
                    self.service.reject(reservation_id, admin, reason)
            except ReservationError as exc:
                return DecisionOutcome(self.service.get(reservation_id), str(exc))
            if action == "approve":
                self._record(reservation_id)
            self._notify(reservation_id)
        return DecisionOutcome(self.service.get(reservation_id))

    def retry_export(self, reservation_id: str) -> Reservation:
        reservation = self.service.get(reservation_id)
        if reservation.status is not Status.APPROVED:
            raise ConflictError("Only approved reservations can be recorded.")
        with self._lock(reservation_id):
            if reservation.export_status is not ExportStatus.RECORDED:
                if self._record(reservation_id):
                    self._notify(reservation_id)
        return self.service.get(reservation_id)

    def recover(self) -> list[str]:
        return []

    def _record(self, reservation_id: str) -> bool:
        result = self.recorder.record(reservation_id)
        if not result.ok:
            self.service.set_export_failed(reservation_id, result.detail)
        return result.ok

    def _notify(self, reservation_id: str) -> None:
        text = self.admin_agent.report_decision(reservation_id)
        self.service.set_notification(reservation_id, text)
