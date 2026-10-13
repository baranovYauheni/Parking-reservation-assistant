"""MCP server (stdio) that records approved reservations to the export file.

Security model:
- the tool accepts only a reservation id and a shared secret; reservation data,
  approval state and the output path come from trusted server-side storage;
- the shared secret is read from configuration or from a 0600 file in the data
  directory (never from the caller), and every call must present it;
- nothing is written unless the reservation is approved in SQL.

Run manually: ``python -m parking_assistant.mcp_server``.
"""

from __future__ import annotations

import hmac
import logging
import os
import sys

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from parking_assistant.config import Settings
from parking_assistant.db import Database
from parking_assistant.export import ExportError, file_lock, write_snapshot
from parking_assistant.reservations import (
    RESERVATION_ID,
    ExportStatus,
    ReservationService,
    Status,
)
from parking_assistant.timeutils import to_utc_text, utc_now

logger = logging.getLogger("parking_assistant.mcp_server")


def record_reservation_core(settings: Settings, reservation_id: str) -> dict:
    """Write the snapshot that includes ``reservation_id`` and mark it recorded."""
    if not RESERVATION_ID.match(reservation_id or ""):
        raise ToolError("invalid reservation id")
    db = Database(settings.db_path)
    service = ReservationService(db)
    with file_lock(settings.export_path):
        try:
            reservation = service.get(reservation_id)
        except Exception as exc:
            raise ToolError("reservation not found") from exc
        if reservation.status is not Status.APPROVED:
            raise ToolError(
                f"reservation is not approved (status={reservation.status})"
            )

        approved = service.list_all(Status.APPROVED)
        try:
            count = write_snapshot(settings.export_path, approved)
        except (OSError, ExportError) as exc:
            raise ToolError(f"export write failed: {type(exc).__name__}") from exc

        with db.transaction() as conn:
            conn.executemany(
                "UPDATE reservations SET export_status = ?, export_error = NULL, "
                "recorded_at = COALESCE(recorded_at, ?) "
                "WHERE id = ? AND status = 'approved'",
                [
                    (ExportStatus.RECORDED.value, to_utc_text(utc_now()), r.id)
                    for r in approved
                ],
            )
    logger.info("recorded reservation %s (snapshot lines=%d)", reservation_id, count)
    return {"status": "recorded", "reservation_id": reservation_id, "lines": count}


def build_server(settings: Settings) -> MCPServer:
    server = MCPServer(
        name="parking-export",
        instructions="Records human-approved parking reservations to the export file.",
    )
    secret = settings.load_mcp_secret()

    @server.tool(
        description=(
            "Record an approved reservation in the export file. Requires the "
            "reservation id and the application auth token. Approval is verified "
            "server-side."
        )
    )
    def record_reservation(reservation_id: str, auth_token: str) -> dict:
        if secret is None or not hmac.compare_digest(
            auth_token.encode(), secret.get_secret_value().encode()
        ):
            raise ToolError("unauthorized")
        return record_reservation_core(settings, reservation_id)

    return server


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    settings = Settings()
    if settings.load_mcp_secret() is None:
        print("MCP secret is not configured", file=sys.stderr)
        raise SystemExit(2)
    if not settings.db_path.exists():
        print(f"database not found: {settings.db_path}", file=sys.stderr)
        raise SystemExit(2)
    os.umask(0o077)
    build_server(settings).run("stdio")


if __name__ == "__main__":
    main()
