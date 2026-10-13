"""RAG evaluation: retrieval Recall@K / Precision@K, answer correctness,
source support, guardrail behaviour and latency (p50/p95).

Usage:
    uv run python scripts/evaluate_rag.py                     # settings from .env
    uv run python scripts/evaluate_rag.py --backend memory --embeddings hashing

Results are written to docs/evaluation/rag-report.md and rag-results.json.
A fresh temporary database is used, so no private data is read.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from parking_assistant.config import PROJECT_ROOT, Settings
from parking_assistant.db import Database
from parking_assistant.embeddings import build_embeddings, describe_embeddings
from parking_assistant.guardrails import check_user_input
from parking_assistant.knowledge import load_chunks
from parking_assistant.llm import build_chat_model
from parking_assistant.rag import RagAnswerer
from parking_assistant.reservations import ReservationService
from parking_assistant.vectorstore import KnowledgeRetriever, build_index

DATASET = PROJECT_ROOT / "data" / "eval" / "rag_questions.jsonl"
DATASET_VERSION = "v1"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--backend", choices=["weaviate", "memory"])
    parser.add_argument("--embeddings", choices=["fastembed", "hashing"])
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "docs" / "evaluation"
    )
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="parking-eval-"))
    overrides = {"PARKING_DATA_DIR": tmp}
    if args.backend:
        overrides["VECTOR_BACKEND"] = args.backend
    if args.embeddings:
        overrides["EMBEDDINGS_PROVIDER"] = args.embeddings
    if args.no_llm:
        overrides["LLM_ENABLED"] = False
    settings = Settings(**overrides)
    settings.prepare_dirs()

    db = Database(settings.db_path)
    db.init()
    service = ReservationService(db)
    index = build_index(settings, build_embeddings(settings))
    index.rebuild(load_chunks(settings.knowledge_dir))
    retriever = KnowledgeRetriever(index=index, k=args.k, min_score=settings.min_score)
    raw_retriever = KnowledgeRetriever(index=index, k=args.k, min_score=-1.0)
    llm = build_chat_model(settings)
    answerer = RagAnswerer(retriever, db, service, llm)

    questions = [json.loads(line) for line in DATASET.read_text().splitlines() if line]
    rows = []
    retrieval_ms: list[float] = []
    total_ms: list[float] = []
    errors = 0

    for q in questions:
        row = {"id": q["id"], "category": q["category"], "question": q["question"]}
        relevant = set(q["relevant_docs"])

        # Retrieval quality is measured without the score threshold (pure top-K).
        started = time.perf_counter()
        top = raw_retriever.invoke(q["question"])
        retrieval_ms.append((time.perf_counter() - started) * 1000)
        ids = [d.metadata["doc_id"] for d in top]
        row["retrieved"] = ids
        if relevant:
            hits = sum(1 for d in ids if d in relevant)
            row["precision_at_k"] = hits / args.k
            row["recall_at_k"] = len(relevant & set(ids)) / len(relevant)
            first = next((i for i, d in enumerate(ids) if d in relevant), None)
            row["reciprocal_rank"] = 0.0 if first is None else 1 / (first + 1)

        for repeat in range(args.repeats):
            started = time.perf_counter()
            try:
                guard = check_user_input(q["question"])
                answer = answerer.answer(q["question"]) if guard.allowed else None
            except Exception as exc:  # noqa: BLE001 - counted as an error
                errors += 1
                row["error"] = repr(exc)
                break
            total_ms.append((time.perf_counter() - started) * 1000)
            if repeat:
                continue
            if q["category"] == "adversarial":
                row["blocked"] = not guard.allowed
                row["correct"] = not guard.allowed
                row["answer"] = guard.message
                continue
            row["answer"] = answer.text
            row["sources"] = [s["doc_id"] for s in answer.sources]
            if q["category"] == "unknown":
                row["correct"] = answer.unknown
                continue
            text = answer.text.lower()
            row["correct"] = all(k.lower() in text for k in q["expected"])
            if q["category"] == "sql":
                row["supported"] = "live-data" in row["sources"]
            else:
                row["supported"] = bool(relevant & set(row["sources"]))
        rows.append(row)

    def mean(key: str, subset: list[dict]) -> float:
        values = [r[key] for r in subset if key in r]
        return sum(values) / len(values) if values else 0.0

    by_cat = {
        c: [r for r in rows if r["category"] == c]
        for c in ("public", "sql", "unknown", "adversarial")
    }
    retrieval_rows = [r for r in rows if "recall_at_k" in r]
    summary = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": f"{DATASET.relative_to(PROJECT_ROOT)} ({DATASET_VERSION})",
        "questions": len(rows),
        "k": args.k,
        "vector_backend": settings.vector_backend,
        "embeddings": describe_embeddings(settings),
        "llm": settings.openai_model
        if llm is not None
        else "offline (extractive answers)",
        "min_score": settings.min_score,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            p: version(p)
            for p in ("langchain", "langgraph", "weaviate-client", "fastembed")
        },
        "retrieval": {
            "evaluated": len(retrieval_rows),
            "recall_at_k": round(mean("recall_at_k", retrieval_rows), 3),
            "precision_at_k": round(mean("precision_at_k", retrieval_rows), 3),
            "mrr": round(mean("reciprocal_rank", retrieval_rows), 3),
        },
        "answers": {
            cat: {
                "n": len(items),
                "correct": sum(1 for r in items if r.get("correct")),
                "supported": sum(1 for r in items if r.get("supported"))
                if cat in ("public", "sql")
                else None,
            }
            for cat, items in by_cat.items()
        },
        "latency_ms": {
            "retrieval_p50": round(percentile(retrieval_ms, 50), 2),
            "retrieval_p95": round(percentile(retrieval_ms, 95), 2),
            "end_to_end_p50": round(percentile(total_ms, 50), 2),
            "end_to_end_p95": round(percentile(total_ms, 95), 2),
            "samples": len(total_ms),
        },
        "errors": errors,
    }
    index.close()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rag-results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2)
    )
    (args.out / "rag-report.md").write_text(render(summary, rows))
    print(json.dumps(summary, indent=2))
    return 0


def render(s: dict, rows: list[dict]) -> str:
    a = s["answers"]
    lines = [
        "# RAG evaluation report",
        "",
        f"Generated by `scripts/evaluate_rag.py` at {s['run_at']}.",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Dataset | {s['dataset']}, {s['questions']} labeled questions |",
        f"| Vector store | {s['vector_backend']} |",
        f"| Embeddings | {s['embeddings']} |",
        f"| Generator | {s['llm']} |",
        f"| K / score threshold | {s['k']} / {s['min_score']} |",
        f"| Python / platform | {s['python']} / {s['platform']} |",
        f"| Library versions | {', '.join(f'{k} {v}' for k, v in s['versions'].items())} |",
        "",
        "## Retrieval",
        "",
        f"Evaluated on {s['retrieval']['evaluated']} questions that have labeled relevant documents "
        "(top-K chunks, no score threshold).",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Recall@{s['k']} | {s['retrieval']['recall_at_k']} |",
        f"| Precision@{s['k']} | {s['retrieval']['precision_at_k']} |",
        f"| MRR | {s['retrieval']['mrr']} |",
        "",
        "Precision@K is computed over chunks: most questions have one relevant document, "
        "so retrieving other (irrelevant) chunks in the remaining slots lowers it even when "
        "the answer is found.",
        "",
        "## Answers",
        "",
        "| Category | Questions | Correct | Supported by source |",
        "| --- | --- | --- | --- |",
    ]
    for cat, v in a.items():
        supported = "-" if v["supported"] is None else v["supported"]
        lines.append(f"| {cat} | {v['n']} | {v['correct']} | {supported} |")
    lat = s["latency_ms"]
    lines += [
        "",
        'Correct = all expected key facts present (public/sql), honest "don\'t know" (unknown), '
        "blocked by guardrails (adversarial). Supported = a cited source is a labeled relevant "
        "document, or live SQL data for SQL questions.",
        "",
        "## Latency",
        "",
        "| Measurement | p50 ms | p95 ms |",
        "| --- | --- | --- |",
        f"| Retrieval | {lat['retrieval_p50']} | {lat['retrieval_p95']} |",
        f"| Guard + retrieval + answer | {lat['end_to_end_p50']} | {lat['end_to_end_p95']} |",
        "",
        f"Samples: {lat['samples']}; errors: {s['errors']}.",
        "",
        "## Per-question results",
        "",
        "| ID | Category | Retrieved (top-K) | Recall | Correct | Supported |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['category']} | {', '.join(r['retrieved'])} | "
            f"{r.get('recall_at_k', '-')} | {r.get('correct', '-')} | {r.get('supported', '-')} |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        "- 28 questions written by the developer from the same documents: this is a development "
        "set, not an independent test set; results are optimistic.",
        "- Keyword-based correctness does not judge fluency or partially correct answers.",
        "- Latency is measured in-process on one machine and excludes network/HTTP and human waiting time.",
        "- With the offline generator answers are extractive; rerun with `OPENAI_API_KEY` set to "
        "evaluate LLM generation.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
