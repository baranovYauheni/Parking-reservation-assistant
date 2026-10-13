"""Wires settings, storage, retrieval, agents and graphs together."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

from langgraph.checkpoint.sqlite import SqliteSaver

from parking_assistant.admin_agent import AdminAgent
from parking_assistant.config import Settings
from parking_assistant.db import Database
from parking_assistant.embeddings import build_embeddings
from parking_assistant.graphs import ChatGraph, ReservationWorkflow
from parking_assistant.knowledge import load_chunks
from parking_assistant.llm import build_chat_model
from parking_assistant.mcp_client import ExportRecorder
from parking_assistant.rag import RagAnswerer
from parking_assistant.reservations import ReservationService
from parking_assistant.timeutils import utc_now
from parking_assistant.vectorstore import (
    KnowledgeIndex,
    KnowledgeRetriever,
    build_index,
)

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    db: Database
    service: ReservationService
    index: KnowledgeIndex
    answerer: RagAnswerer
    admin_agent: AdminAgent
    recorder: ExportRecorder
    workflow: ReservationWorkflow
    chat: ChatGraph
    checkpoint_conn: sqlite3.Connection

    def close(self) -> None:
        self.index.close()
        self.checkpoint_conn.close()


def build_container(
    settings: Settings,
    *,
    clock=utc_now,
    index: KnowledgeIndex | None = None,
    recorder: ExportRecorder | None = None,
    llm=None,
    use_configured_llm: bool = True,
) -> Container:
    settings.prepare_dirs()
    settings.ensure_secrets()

    db = Database(settings.db_path)
    db.init()
    service = ReservationService(db, clock=clock)

    if index is None:
        index = build_index(settings, build_embeddings(settings))
    chunks = load_chunks(settings.knowledge_dir)
    index.rebuild(chunks)
    logger.info("indexed %d public knowledge chunks", len(chunks))

    if llm is None and use_configured_llm:
        llm = build_chat_model(settings)
    retriever = KnowledgeRetriever(
        index=index, k=settings.retrieval_k, min_score=settings.min_score
    )
    answerer = RagAnswerer(retriever, db, service, llm)
    admin_agent = AdminAgent(service, llm)
    recorder = recorder or ExportRecorder(settings)

    conn = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode = WAL")
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    workflow = ReservationWorkflow(service, admin_agent, recorder, checkpointer)
    chat = ChatGraph(service, answerer, workflow, checkpointer)
    return Container(
        settings=settings,
        db=db,
        service=service,
        index=index,
        answerer=answerer,
        admin_agent=admin_agent,
        recorder=recorder,
        workflow=workflow,
        chat=chat,
        checkpoint_conn=conn,
    )
