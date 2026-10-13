"""Wires settings, storage, retrieval and the chat agent together."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from parking_assistant.chat import ChatService
from parking_assistant.config import Settings
from parking_assistant.db import Database
from parking_assistant.embeddings import build_embeddings
from parking_assistant.knowledge import load_chunks
from parking_assistant.llm import build_chat_model
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
    chat: ChatService

    def close(self) -> None:
        self.index.close()


def build_container(
    settings: Settings,
    *,
    clock=utc_now,
    index: KnowledgeIndex | None = None,
    llm=None,
    use_configured_llm: bool = True,
) -> Container:
    settings.prepare_dirs()

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
    chat = ChatService(db, service, answerer)
    return Container(
        settings=settings,
        db=db,
        service=service,
        index=index,
        answerer=answerer,
        chat=chat,
    )
