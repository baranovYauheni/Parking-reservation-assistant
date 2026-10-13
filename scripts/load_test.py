"""Controlled load test of the three components with synthetic data:

1. chat in interactive dialogue mode (info questions + full booking dialog) via HTTP;
2. administrator approvals via the admin API (all requests compete for the same
   period, so only ``capacity`` can be approved);
3. MCP recording (repeated record calls for the approved reservations).

Integrity checks: no overbooking, export lines == approved reservations, no
duplicate lines. Results: docs/evaluation/load-report.md.

    uv run python scripts/load_test.py --sessions 20 --workers 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from parking_assistant.api import create_app
from parking_assistant.config import PROJECT_ROOT, Settings
from parking_assistant.timeutils import to_local

ADMIN_TOKEN = "load-test-admin-token"


def pct(values: list[float], q: int) -> float:
    if len(values) < 2:
        return round(values[0], 1) if values else 0.0
    return round(statistics.quantiles(values, n=100, method="inclusive")[q - 1], 1)


def stats(values: list[float]) -> dict:
    return {
        "n": len(values),
        "p50": pct(values, 50),
        "p95": pct(values, 95),
        "max": round(max(values or [0]), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--admin-workers", type=int, default=4)
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "docs" / "evaluation"
    )
    args = parser.parse_args()

    data_dir = Path(tempfile.mkdtemp(prefix="parking-load-"))
    settings = Settings(
        PARKING_DATA_DIR=data_dir,
        VECTOR_BACKEND="memory",
        EMBEDDINGS_PROVIDER="hashing",
        LLM_ENABLED=False,
        ADMIN_TOKEN=ADMIN_TOKEN,
    )
    app = create_app(settings)
    start_local = to_local(datetime.now(UTC) + timedelta(days=2)).strftime(
        "%Y-%m-%d 10:00"
    )
    end_local = start_local.replace("10:00", "13:00")

    chat_latency: list[float] = []
    chat_errors: list[str] = []

    def session(i: int) -> None:
        client = TestClient(app)
        messages = [
            "Where is the parking located?",
            "How much does parking cost?",
            "I want to book a space",
            f"name: Test User{chr(65 + i % 26)}, car: LT{i:05d}",
            f"from: {start_local}, to: {end_local}",
            "confirm",
            "status",
        ]
        for text in messages:
            started = time.perf_counter()
            response = client.post("/api/chat", json={"message": text})
            chat_latency.append((time.perf_counter() - started) * 1000)
            if response.status_code != 200:
                chat_errors.append(f"{response.status_code} {text}")
            elif text == "confirm" and response.json()["route"] != "submitted":
                chat_errors.append(f"not submitted: {response.json()['reply'][:80]}")

    with TestClient(app) as admin:
        headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
        capacity = admin.get("/api/facility").json()["total_spaces"]

        t0 = time.perf_counter()
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(session, range(args.sessions)))
        chat_wall = time.perf_counter() - t0

        pending = admin.get(
            "/api/admin/requests?status=pending", headers=headers
        ).json()["requests"]
        decision_latency: list[float] = []
        outcomes: Counter[int] = Counter()

        def approve(rid: str) -> None:
            started = time.perf_counter()
            response = admin.post(
                f"/api/admin/requests/{rid}/decision",
                json={"action": "approve"},
                headers=headers,
            )
            decision_latency.append((time.perf_counter() - started) * 1000)
            outcomes[response.status_code] += 1

        t0 = time.perf_counter()
        with ThreadPoolExecutor(args.admin_workers) as pool:
            list(pool.map(approve, [r["id"] for r in pending]))
        admin_wall = time.perf_counter() - t0

        approved = admin.get(
            "/api/admin/requests?status=approved", headers=headers
        ).json()["requests"]
        container = app.state.container
        mcp_latency: list[float] = []
        mcp_fail = 0

        def record(rid: str) -> None:
            nonlocal mcp_fail
            started = time.perf_counter()
            result = container.recorder.record(rid)
            mcp_latency.append((time.perf_counter() - started) * 1000)
            if not result.ok:
                mcp_fail += 1

        t0 = time.perf_counter()
        with ThreadPoolExecutor(args.admin_workers) as pool:
            list(pool.map(record, [r["id"] for r in approved] * 2))
        mcp_wall = time.perf_counter() - t0

        lines = admin.get("/api/admin/export", headers=headers).json()["lines"]

    integrity = {
        "submitted_requests": len(pending),
        "capacity": capacity,
        "approved": len(approved),
        "no_overbooking": len(approved) <= capacity,
        "export_lines": len(lines),
        "export_matches_approved": len(lines) == len(approved),
        "no_duplicate_lines": len(lines) == len(set(lines)),
        "all_recorded": all(r["export_status"] == "recorded" for r in approved),
    }
    result = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "sessions": args.sessions,
            "chat_workers": args.workers,
            "admin_workers": args.admin_workers,
            "vector_backend": "memory",
            "embeddings": "hashing",
            "llm": "offline",
        },
        "chat": stats(chat_latency)
        | {
            "errors": len(chat_errors),
            "wall_s": round(chat_wall, 2),
            "throughput_msg_s": round(len(chat_latency) / chat_wall, 1),
        },
        "admin_decisions": stats(decision_latency)
        | {"status_codes": dict(outcomes), "wall_s": round(admin_wall, 2)},
        "mcp_record": stats(mcp_latency)
        | {"failures": mcp_fail, "wall_s": round(mcp_wall, 2)},
        "integrity": integrity,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "load-results.json").write_text(json.dumps(result, indent=2))
    (args.out / "load-report.md").write_text(render(result))
    print(json.dumps(result, indent=2))
    return 0 if all(v for k, v in integrity.items() if isinstance(v, bool)) else 1


def render(r: dict) -> str:
    c, a, m, i = r["chat"], r["admin_decisions"], r["mcp_record"], r["integrity"]
    return "\n".join(
        [
            "# Load test report",
            "",
            f"Generated by `scripts/load_test.py` at {r['run_at']} (synthetic data, single process, "
            "single uvicorn-equivalent worker via TestClient, offline LLM, in-memory vector store).",
            "",
            "| Component | Requests | p50 ms | p95 ms | max ms | Errors | Wall s |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            f"| Chat (HTTP, {r['config']['sessions']} sessions x 7 msgs, {r['config']['chat_workers']} threads) | {c['n']} | {c['p50']} | {c['p95']} | {c['max']} | {c['errors']} | {c['wall_s']} |",
            f"| Admin decision incl. MCP recording ({r['config']['admin_workers']} threads) | {a['n']} | {a['p50']} | {a['p95']} | {a['max']} | {a['status_codes']} | {a['wall_s']} |",
            f"| MCP record (repeat, idempotent) | {m['n']} | {m['p50']} | {m['p95']} | {m['max']} | {m['failures']} | {m['wall_s']} |",
            "",
            "## Integrity",
            "",
            "| Check | Result |",
            "| --- | --- |",
            *[f"| {k} | {v} |" for k, v in i.items()],
            "",
            "All sessions request the same period, so exactly `capacity` approvals succeed and the "
            "rest return HTTP 409 (no free space) - this verifies the transactional capacity check "
            "under concurrency. Each MCP call spawns a stdio server process, which dominates its latency.",
            "",
            "Limitations: one machine, in-process HTTP client, no real LLM latency; numbers are a "
            "baseline for this demo, not production capacity.",
            "",
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
