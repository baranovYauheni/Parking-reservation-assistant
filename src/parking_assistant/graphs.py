"""LangGraph orchestration.

Two persistent graphs share one SQLite checkpointer:

* **chat graph** (thread ``chat:<session>``) - one run per user message:
  guard -> route -> info (RAG) | booking dialog | status -> submit.
  ``submit`` creates the pending request and starts the reservation workflow.

* **reservation workflow** (thread ``reservation:<id>``) - one run per request:
  escalate (admin agent) -> await_decision (interrupt, human) -> apply_decision
  (SQL transaction) -> record (MCP) -> notify (admin agent).
  It pauses at ``interrupt`` until the admin API resumes it, survives process
  restarts through the checkpointer, and pauses at ``await_retry`` when the MCP
  recording fails.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from parking_assistant.admin_agent import AdminAgent
from parking_assistant.booking_dialog import (
    current_prompt,
    handle_booking_message,
    is_question,
    new_booking_state,
    start_booking,
    wants_booking,
    wants_status,
)
from parking_assistant.guardrails import check_user_input
from parking_assistant.mcp_client import ExportRecorder
from parking_assistant.rag import RagAnswerer
from parking_assistant.reservations import (
    ConflictError,
    ExportStatus,
    Reservation,
    ReservationError,
    ReservationRequest,
    ReservationService,
    Status,
    describe_status,
)
from parking_assistant.timeutils import human_local

logger = logging.getLogger(__name__)

MAX_HISTORY = 40


# ============================================================================
# Reservation workflow
# ============================================================================


class WorkflowState(TypedDict, total=False):
    reservation_id: str
    decision: dict
    decision_error: str
    export_error: str
    attempts: int
    notification: str


@dataclass
class DecisionOutcome:
    reservation: Reservation
    error: str | None = None


class ReservationWorkflow:
    def __init__(
        self,
        service: ReservationService,
        admin_agent: AdminAgent,
        recorder: ExportRecorder,
        checkpointer: Any,
    ) -> None:
        self.service = service
        self.admin_agent = admin_agent
        self.recorder = recorder
        self._locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()
        self.graph = self._build().compile(checkpointer=checkpointer)

    # ---- nodes ---------------------------------------------------------------

    def _escalate(self, state: WorkflowState) -> WorkflowState:
        self.admin_agent.escalate(state["reservation_id"])
        return {}

    def _await_decision(self, state: WorkflowState) -> WorkflowState:
        decision = interrupt(
            {"type": "admin_decision", "reservation_id": state["reservation_id"]}
        )
        return {"decision": decision, "decision_error": ""}

    def _apply_decision(self, state: WorkflowState) -> WorkflowState:
        rid = state["reservation_id"]
        decision = state["decision"]
        try:
            if decision["action"] == "approve":
                self.service.approve(rid, decision["admin"])
            else:
                self.service.reject(rid, decision["admin"], decision.get("reason", ""))
        except ReservationError as exc:
            return {"decision_error": str(exc)}
        return {"decision_error": ""}

    def _route_after_decision(
        self, state: WorkflowState
    ) -> Literal["await_decision", "record", "notify"]:
        reservation = self.service.get(state["reservation_id"])
        if reservation.status is Status.PENDING:
            return "await_decision"
        if reservation.status is Status.APPROVED:
            return "record"
        return "notify"

    def _record(self, state: WorkflowState) -> WorkflowState:
        rid = state["reservation_id"]
        attempts = state.get("attempts", 0) + 1
        result = self.recorder.record(rid)
        if result.ok:
            return {"export_error": "", "attempts": attempts}
        self.service.set_export_failed(rid, result.detail)
        return {"export_error": result.detail, "attempts": attempts}

    def _route_after_record(
        self, state: WorkflowState
    ) -> Literal["notify", "await_retry"]:
        reservation = self.service.get(state["reservation_id"])
        if reservation.export_status is ExportStatus.RECORDED:
            return "notify"
        return "await_retry"

    def _await_retry(self, state: WorkflowState) -> WorkflowState:
        interrupt({"type": "export_retry", "reservation_id": state["reservation_id"]})
        return {}

    def _notify(self, state: WorkflowState) -> WorkflowState:
        rid = state["reservation_id"]
        text = self.admin_agent.report_decision(rid)
        self.service.set_notification(rid, text)
        return {"notification": text}

    def _build(self) -> StateGraph:
        graph = StateGraph(WorkflowState)
        graph.add_node("escalate", self._escalate)
        graph.add_node("await_decision", self._await_decision)
        graph.add_node("apply_decision", self._apply_decision)
        graph.add_node("record", self._record)
        graph.add_node("await_retry", self._await_retry)
        graph.add_node("notify", self._notify)
        graph.add_edge(START, "escalate")
        graph.add_edge("escalate", "await_decision")
        graph.add_edge("await_decision", "apply_decision")
        graph.add_conditional_edges("apply_decision", self._route_after_decision)
        graph.add_conditional_edges("record", self._route_after_record)
        graph.add_edge("await_retry", "record")
        graph.add_edge("notify", END)
        return graph

    # ---- public API ----------------------------------------------------------

    @staticmethod
    def config(reservation_id: str) -> dict:
        return {"configurable": {"thread_id": f"reservation:{reservation_id}"}}

    def _lock(self, reservation_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks[reservation_id]

    def waiting_for(self, reservation_id: str) -> str | None:
        """Type of the interrupt the workflow is paused at, if any."""
        snapshot = self.graph.get_state(self.config(reservation_id))
        for item in snapshot.interrupts:
            return item.value.get("type")
        return None

    def start(self, reservation_id: str) -> None:
        with self._lock(reservation_id):
            snapshot = self.graph.get_state(self.config(reservation_id))
            if snapshot.values:
                return
            self.graph.invoke(
                {"reservation_id": reservation_id, "attempts": 0},
                self.config(reservation_id),
            )

    def decide(
        self,
        reservation_id: str,
        action: Literal["approve", "reject"],
        admin: str,
        reason: str = "",
    ) -> DecisionOutcome:
        current = self.service.get(reservation_id)
        with self._lock(reservation_id):
            cfg = self.config(reservation_id)
            snapshot = self.graph.get_state(cfg)
            if not snapshot.values:
                # Checkpoint missing (e.g. created before a reset): start it now.
                self.graph.invoke(
                    {"reservation_id": reservation_id, "attempts": 0}, cfg
                )
            if self.waiting_for(reservation_id) != "admin_decision":
                # Already decided: repeating the same decision is idempotent.
                same = (action == "approve" and current.status is Status.APPROVED) or (
                    action == "reject" and current.status is Status.REJECTED
                )
                if same:
                    return DecisionOutcome(current)
                raise ConflictError(f"The request is already {current.status.value}.")
            self.graph.invoke(
                Command(resume={"action": action, "admin": admin, "reason": reason}),
                cfg,
            )
            state = self.graph.get_state(cfg).values
        error = state.get("decision_error") or None
        return DecisionOutcome(self.service.get(reservation_id), error)

    def retry_export(self, reservation_id: str) -> Reservation:
        reservation = self.service.get(reservation_id)
        if reservation.status is not Status.APPROVED:
            raise ConflictError("Only approved reservations can be recorded.")
        with self._lock(reservation_id):
            cfg = self.config(reservation_id)
            if self.waiting_for(reservation_id) == "export_retry":
                self.graph.invoke(Command(resume="retry"), cfg)
            elif reservation.export_status is not ExportStatus.RECORDED:
                result = self.recorder.record(reservation_id)
                if not result.ok:
                    self.service.set_export_failed(reservation_id, result.detail)
        return self.service.get(reservation_id)

    def recover(self) -> list[str]:
        """Resume workflows that were interrupted by a crash or restart."""
        resumed = []
        for reservation in self.service.list_all():
            cfg = self.config(reservation.id)
            with self._lock(reservation.id):
                snapshot = self.graph.get_state(cfg)
                if not snapshot.values:
                    if reservation.status is Status.PENDING:
                        self.graph.invoke(
                            {"reservation_id": reservation.id, "attempts": 0}, cfg
                        )
                        resumed.append(reservation.id)
                    continue
                if snapshot.next and not snapshot.interrupts:
                    self.graph.invoke(None, cfg)
                    resumed.append(reservation.id)
                elif (
                    snapshot.interrupts
                    and snapshot.interrupts[0].value.get("type") == "export_retry"
                    and reservation.export_status is ExportStatus.RECORDED
                ):
                    self.graph.invoke(Command(resume="retry"), cfg)
                    resumed.append(reservation.id)
        return resumed


# ============================================================================
# Chat graph
# ============================================================================


class ChatState(TypedDict, total=False):
    session_id: str
    input: str
    reply: str
    sources: list[dict]
    route: str
    booking: dict
    pending_request: dict | None
    history: list[dict]
    submitted: list[str]


class ChatGraph:
    def __init__(
        self,
        service: ReservationService,
        answerer: RagAnswerer,
        workflow: ReservationWorkflow,
        checkpointer: Any,
    ) -> None:
        self.service = service
        self.answerer = answerer
        self.workflow = workflow
        self._locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()
        self.graph = self._build().compile(checkpointer=checkpointer)

    # ---- nodes ---------------------------------------------------------------

    def _guard(self, state: ChatState) -> ChatState:
        result = check_user_input(state["input"])
        if result.allowed:
            return {"route": "", "reply": "", "sources": [], "pending_request": None}
        logger.info("input blocked: %s", result.category)
        return {
            "route": "blocked",
            "reply": result.message,
            "sources": [],
            "pending_request": None,
        }

    def _route(self, state: ChatState) -> str:
        if state.get("route") == "blocked":
            return "finish"
        text = state["input"]
        booking = state.get("booking") or new_booking_state()
        in_booking = booking["mode"] != "idle"
        if in_booking:
            if is_question(text) and not wants_booking(text):
                return "info"
            return "booking"
        if wants_status(text):
            return "status"
        if wants_booking(text):
            return "booking"
        return "info"

    def _info(self, state: ChatState) -> ChatState:
        answer = self.answerer.answer(state["input"])
        reply = answer.text
        booking = state.get("booking") or new_booking_state()
        if booking["mode"] != "idle":
            reply += "\n\nBack to your reservation: " + current_prompt(
                dict(booking), self.service
            )
        return {"reply": reply, "sources": answer.sources, "route": "info"}

    def _booking(self, state: ChatState) -> ChatState:
        booking = state.get("booking") or new_booking_state()
        if booking["mode"] == "idle":
            step = start_booking(booking, state["input"], self.service)
        else:
            step = handle_booking_message(booking, state["input"], self.service)
        pending = step.submit.model_dump(mode="json") if step.submit else None
        return {
            "reply": step.reply,
            "booking": step.state,
            "pending_request": pending,
            "route": "booking",
        }

    def _after_booking(self, state: ChatState) -> str:
        return "submit" if state.get("pending_request") else "finish"

    def _submit(self, state: ChatState) -> ChatState:
        request = ReservationRequest.model_validate(state["pending_request"])
        try:
            reservation = self.service.submit(state["session_id"], request)
        except ReservationError as exc:
            booking = dict(state["booking"])
            return {
                "reply": f"I couldn't submit the request: {exc} "
                + current_prompt(booking, self.service),
                "booking": booking,
                "pending_request": None,
            }
        try:
            self.workflow.start(reservation.id)
            status_text = (
                "It is now waiting for a human administrator's decision. "
                "Ask me 'status' at any time to check it."
            )
        except Exception:
            logger.exception("workflow start failed for %s", reservation.id)
            status_text = (
                "It was saved but could not be forwarded right now; it will be "
                "forwarded automatically."
            )
        submitted = list(state.get("submitted") or []) + [reservation.id]
        return {
            "reply": f"Your reservation request #{reservation.id[:8]} was submitted. "
            + status_text,
            "booking": new_booking_state(),
            "pending_request": None,
            "submitted": submitted,
            "route": "submitted",
        }

    def _status(self, state: ChatState) -> ChatState:
        reservations = self.service.list_for_session(state["session_id"])
        if not reservations:
            return {
                "reply": "You have no reservation requests in this chat yet. "
                "Say 'book' to make one.",
                "route": "status",
            }
        lines = ["Your reservation requests:"]
        for r in reservations:
            lines.append(
                f"- #{r.id[:8]} {r.car_number}, {human_local(r.start)} - "
                f"{human_local(r.end)}: {describe_status(r)}"
            )
        return {"reply": "\n".join(lines), "route": "status"}

    def _finish(self, state: ChatState) -> ChatState:
        history = list(state.get("history") or [])
        history.append({"role": "user", "content": state["input"]})
        history.append(
            {
                "role": "assistant",
                "content": state["reply"],
                "sources": state.get("sources") or [],
            }
        )
        return {"history": history[-MAX_HISTORY:]}

    def _build(self) -> StateGraph:
        graph = StateGraph(ChatState)
        graph.add_node("guard", self._guard)
        graph.add_node("info", self._info)
        graph.add_node("booking", self._booking)
        graph.add_node("submit", self._submit)
        graph.add_node("status", self._status)
        graph.add_node("finish", self._finish)
        graph.add_edge(START, "guard")
        graph.add_conditional_edges(
            "guard",
            self._route,
            {
                "info": "info",
                "booking": "booking",
                "status": "status",
                "finish": "finish",
            },
        )
        graph.add_conditional_edges(
            "booking", self._after_booking, {"submit": "submit", "finish": "finish"}
        )
        graph.add_edge("info", "finish")
        graph.add_edge("submit", "finish")
        graph.add_edge("status", "finish")
        graph.add_edge("finish", END)
        return graph

    # ---- public API ------------------------------------------------------------

    @staticmethod
    def config(session_id: str) -> dict:
        return {"configurable": {"thread_id": f"chat:{session_id}"}}

    def send(self, session_id: str, text: str) -> ChatState:
        with self._locks_guard:
            lock = self._locks[session_id]
        with lock:
            return self.graph.invoke(
                {"session_id": session_id, "input": text.strip()},
                self.config(session_id),
            )

    def history(self, session_id: str) -> list[dict]:
        return self.graph.get_state(self.config(session_id)).values.get("history", [])
