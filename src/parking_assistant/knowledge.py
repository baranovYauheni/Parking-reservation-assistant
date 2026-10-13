"""Loading and chunking of the public knowledge base.

Only allow-listed files are indexed, and every chunk passes a sensitive-content
check before it can reach the vector store.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

from parking_assistant.guardrails import find_sensitive

PUBLIC_DOCUMENTS: tuple[str, ...] = (
    "overview.md",
    "location.md",
    "booking-process.md",
    "pricing-policy.md",
    "rules.md",
    "services.md",
    "faq.md",
    "privacy.md",
)

_FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


class KnowledgeError(Exception):
    pass


def parse_document(text: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        raise KnowledgeError("document has no front matter")
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    if "id" not in meta or "title" not in meta:
        raise KnowledgeError("front matter must contain id and title")
    return meta, text[match.end() :]


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown into ``(heading, text)`` pairs on ``##`` headings."""
    sections: list[tuple[str, str]] = []
    heading = ""
    lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if "".join(lines).strip():
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = line[3:].strip(), []
        else:
            lines.append(line)
    if "".join(lines).strip():
        sections.append((heading, "\n".join(lines).strip()))
    return sections


def load_chunks(
    knowledge_dir: Path, allowlist: tuple[str, ...] = PUBLIC_DOCUMENTS
) -> list[Document]:
    chunks: list[Document] = []
    for filename in allowlist:
        path = knowledge_dir / filename
        meta, body = parse_document(path.read_text(encoding="utf-8"))
        for index, (section, text) in enumerate(split_sections(body)):
            content = f"{meta['title']} - {section}\n{text}" if section else text
            found = find_sensitive(content)
            if found:
                raise KnowledgeError(
                    f"{filename} section {section!r} contains sensitive data: {found}"
                )
            chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        "doc_id": meta["id"],
                        "title": meta["title"],
                        "section": section,
                        "source": filename,
                        "chunk_id": f"{meta['id']}#{index}",
                    },
                )
            )
    return chunks
