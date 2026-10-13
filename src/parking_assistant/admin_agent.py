"""Second agent: a LangChain tool-calling agent that talks to the admin queue.

The agent can only (1) put a request into the human review queue and
(2) read a request's status. It has no tool that approves or rejects: the
decision is made by a human through the authenticated admin API. The model
sees only the reservation id and status, never personal data.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from parking_assistant.reservations import ReservationService, Status

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the administrator-liaison agent of a parking garage. You forward "
    "reservation requests to a HUMAN administrator and report the human's "
    "decision. You never approve or reject anything yourself and never invent "
    "a status: always use the tools. Use only the reservation id you are given. "
    "Answer in one or two short sentences addressed to the visitor."
)

_ID = re.compile(r"\b[0-9a-f]{32}\b")


class OfflineAdminModel(BaseChatModel):
    """Deterministic chat model used when no LLM is configured.

    It behaves like a tool-calling LLM: first it calls the bound tool with the
    reservation id from the task, then it turns the tool result into a reply.
    """

    tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "offline-admin-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> OfflineAdminModel:
        names = [getattr(t, "name", None) or t.get("name") for t in tools]
        return self.model_copy(update={"tool_names": names})

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs
    ) -> ChatResult:
        last = messages[-1]
        if isinstance(last, ToolMessage):
            message = AIMessage(content=_describe(str(last.content)))
        else:
            task = next(
                (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
                "",
            )
            match = _ID.search(str(task))
            if not match or not self.tool_names:
                message = AIMessage(content="I could not identify the request.")
            else:
                message = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": self.tool_names[0],
                            "args": {"reservation_id": match.group(0)},
                            "id": "call_offline_1",
                            "type": "tool_call",
                        }
                    ],
                )
        return ChatResult(generations=[ChatGeneration(message=message)])


def _describe(tool_result: str) -> str:
    if "status=approved" in tool_result:
        return "Good news: your reservation was approved."
    if "status=rejected" in tool_result:
        return "Unfortunately the administrator rejected your reservation request."
    if "queued" in tool_result:
        return "Your request is now waiting for a human administrator's decision."
    return f"Current state: {tool_result}"


def _status_line(service: ReservationService, reservation_id: str) -> str:
    r = service.get(reservation_id)
    return f"request={r.id[:8]} status={r.status.value}"


class AdminAgent:
    def __init__(
        self, service: ReservationService, model: BaseChatModel | None = None
    ) -> None:
        self.service = service
        self.model = model or OfflineAdminModel()

    def _run(self, tools: list, task: str) -> str:
        agent = create_agent(self.model, tools=tools, system_prompt=SYSTEM_PROMPT)
        result = agent.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={"recursion_limit": 8},
        )
        final = result["messages"][-1]
        return str(final.content).strip()

    def escalate(self, reservation_id: str) -> str:
        """Ask the agent to enqueue the request for human review."""
        service = self.service

        @tool
        def enqueue_for_human_review(reservation_id: str) -> str:
            """Put a confirmed reservation request into the human administrator's
            review queue. Returns the queue status."""
            if reservation_id != allowed_id:
                return "error: not allowed for this request"
            reservation = service.mark_queued(reservation_id)
            return f"queued; {_status_line(service, reservation.id)}"

        allowed_id = reservation_id
        task = (
            f"A visitor confirmed reservation request {reservation_id}. Put it into "
            "the human review queue and tell the visitor it is waiting for approval."
        )
        try:
            message = self._run([enqueue_for_human_review], task)
        except Exception as exc:
            logger.warning("admin agent escalate failed: %s", exc)
            message = ""
        # The queue entry must exist regardless of what the model did.
        if self.service.get(reservation_id).queued_at is None:
            logger.warning(
                "agent did not enqueue %s; enqueuing directly", reservation_id
            )
            self.service.mark_queued(reservation_id)
        return message or "Your request is waiting for a human administrator."

    def report_decision(self, reservation_id: str) -> str:
        """Ask the agent to turn the human decision into a visitor message."""
        service = self.service

        @tool
        def get_request_status(reservation_id: str) -> str:
            """Read the current status of a reservation request (no personal data)."""
            if reservation_id != allowed_id:
                return "error: not allowed for this request"
            return _status_line(service, reservation_id)

        allowed_id = reservation_id
        task = (
            f"The human administrator has decided on request {reservation_id}. "
            "Look up its status and write a short message for the visitor."
        )
        reservation = self.service.get(reservation_id)
        fallback = _describe(f"status={reservation.status.value}")
        try:
            message = self._run([get_request_status], task)
        except Exception as exc:
            logger.warning("admin agent report failed: %s", exc)
            return fallback
        lowered = message.lower()
        if reservation.status is Status.REJECTED and "approved" in lowered:
            return fallback
        return message or fallback
