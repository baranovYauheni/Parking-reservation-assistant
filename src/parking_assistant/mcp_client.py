"""MCP client used by the workflow to record approved reservations."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

import anyio
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from parking_assistant.config import PROJECT_ROOT, Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordResult:
    ok: bool
    detail: str


class ExportRecorder:
    """Spawns the MCP server over stdio and calls ``record_reservation``."""

    def __init__(self, settings: Settings, command: list[str] | None = None) -> None:
        self.settings = settings
        self.command = command or [sys.executable, "-m", "parking_assistant.mcp_server"]

    def _params(self) -> StdioServerParameters:
        env = {
            "PARKING_DATA_DIR": str(self.settings.data_dir),
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
        }
        return StdioServerParameters(
            command=self.command[0], args=self.command[1:], env=env, cwd=PROJECT_ROOT
        )

    async def _call(self, reservation_id: str) -> RecordResult:
        secret = self.settings.mcp_shared_secret
        with anyio.fail_after(self.settings.mcp_timeout_seconds):
            async with Client(self._params()) as client:
                result = await client.call_tool(
                    "record_reservation",
                    {
                        "reservation_id": reservation_id,
                        "auth_token": secret.get_secret_value() if secret else "",
                    },
                )
        text = " ".join(
            getattr(block, "text", "") for block in (result.content or [])
        ).strip()
        if result.is_error:
            return RecordResult(False, text or "MCP tool error")
        return RecordResult(True, text or "recorded")

    def record(self, reservation_id: str) -> RecordResult:
        try:
            return anyio.run(self._call, reservation_id)
        except Exception as exc:  # transport failure, timeout, spawn error
            logger.warning("MCP call failed for %s: %s", reservation_id, exc)
            return RecordResult(False, f"MCP server unavailable: {type(exc).__name__}")
