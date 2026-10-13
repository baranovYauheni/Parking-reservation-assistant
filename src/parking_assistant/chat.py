"""Visitor chat agent: guard -> route -> info (RAG) | booking dialog | status.

The per-session conversation state (booking draft, history) is stored as JSON
in SQLite, so drafts survive page reloads and restarts.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from typing import Any

from parking_assistant.booking_dialog import (
    current_prompt,
    handle_booking_message,
    is_question,
    new_booking_state,
    start_booking,
    wants_booking,
    wants_status,
)
from parking_assistant.db import Database
from parking_assistant.guardrails import check_user_input
from parking_assistant.rag import RagAnswerer
from parking_assistant.reservations import (
    ReservationError,
    ReservationService,
    describe_status,
)
from parking_assistant.timeutils import human_local, to_utc_text

logger = logging.getLogger(__name__)

MAX_HISTORY = 40

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    state      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class ChatService:
    def __init__(
        self,
        db: Database,
        service: ReservationService,
        answerer: RagAnswerer,
        workflow: Any = None,
    ) -> None:
        self.db = db
        self.service = service
        self.answerer = answerer
        self.workflow = workflow
        self._locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()
        with self.db.connection() as conn:
            conn.executescript(SCHEMA)

    # ---- state persistence -----------------------------------------------------

    def _load(self, session_id: str) -> dict:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT state FROM chat_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return {"booking": new_booking_state(), "history": [], "submitted": []}
        return json.loads(row["state"])

    def _save(self, session_id: str, state: dict) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO chat_sessions (session_id, state, updated_at) "
                "VALUES (?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET "
                "state = excluded.state, updated_at = excluded.updated_at",
                (session_id, json.dumps(state), to_utc_text(self.service.clock())),
            )

    # ---- public API ------------------------------------------------------------

    def send(self, session_id: str, text: str) -> dict:
        with self._locks_guard:
            lock = self._locks[session_id]
        with lock:
            state = self._load(session_id)
            result = self._handle(session_id, state, text.strip())
            history = state["history"]
            history.append({"role": "user", "content": text.strip()})
            history.append(
                {
                    "role": "assistant",
                    "content": result["reply"],
                    "sources": result.get("sources", []),
                }
            )
            state["history"] = history[-MAX_HISTORY:]
            self._save(session_id, state)
            return result | {"booking": state["booking"]}

    def history(self, session_id: str) -> list[dict]:
        return self._load(session_id)["history"]

    # ---- steps -------------------------------------------------------------------

    def _handle(self, session_id: str, state: dict, text: str) -> dict:
        guard = check_user_input(text)
        if not guard.allowed:
            logger.info("input blocked: %s", guard.category)
            return {"reply": guard.message, "sources": [], "route": "blocked"}

        booking = state["booking"]
        if booking["mode"] != "idle":
            if is_question(text) and not wants_booking(text):
                return self._info(text, booking)
            return self._booking(session_id, state, text)
        if wants_status(text):
            return self._status(session_id)
        if wants_booking(text):
            return self._booking(session_id, state, text)
        return self._info(text, booking)

    def _info(self, text: str, booking: dict) -> dict:
        answer = self.answerer.answer(text)
        reply = answer.text
        if booking["mode"] != "idle":
            reply += "\n\nBack to your reservation: " + current_prompt(
                dict(booking), self.service
            )
        return {"reply": reply, "sources": answer.sources, "route": "info"}

    def _booking(self, session_id: str, state: dict, text: str) -> dict:
        booking = state["booking"]
        if booking["mode"] == "idle":
            step = start_booking(booking, text, self.service)
        else:
            step = handle_booking_message(booking, text, self.service)
        state["booking"] = step.state
        if step.submit is None:
            return {"reply": step.reply, "sources": [], "route": "booking"}

        try:
            reservation = self.service.submit(session_id, step.submit)
        except ReservationError as exc:
            reply = f"I couldn't submit the request: {exc} " + current_prompt(
                step.state, self.service
            )
            return {"reply": reply, "sources": [], "route": "booking"}

        status_text = "It is pending administrator approval."
        if self.workflow is not None:
            try:
                self.workflow.start(reservation.id)
                status_text = (
                    "It is now waiting for a human administrator's decision. "
                    "Ask me 'status' at any time to check it."
                )
            except Exception:
                logger.exception("escalation failed for %s", reservation.id)
                status_text = "It was saved but could not be forwarded right now."
        state["booking"] = new_booking_state()
        state["submitted"] = state.get("submitted", []) + [reservation.id]
        return {
            "reply": f"Your reservation request #{reservation.id[:8]} was submitted. "
            + status_text,
            "sources": [],
            "route": "submitted",
            "submitted": state["submitted"],
        }

    def _status(self, session_id: str) -> dict:
        reservations = self.service.list_for_session(session_id)
        if not reservations:
            return {
                "reply": "You have no reservation requests in this chat yet. "
                "Say 'book' to make one.",
                "sources": [],
                "route": "status",
            }
        lines = ["Your reservation requests:"]
        for r in reservations:
            lines.append(
                f"- #{r.id[:8]} {r.car_number}, {human_local(r.start)} - "
                f"{human_local(r.end)}: {describe_status(r)}"
            )
        return {"reply": "\n".join(lines), "sources": [], "route": "status"}
