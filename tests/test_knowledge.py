import pytest

from parking_assistant.knowledge import (
    PUBLIC_DOCUMENTS,
    KnowledgeError,
    load_chunks,
    parse_document,
    split_sections,
)
from tests.conftest import KNOWLEDGE_DIR


def test_all_public_documents_are_chunked_with_source_metadata():
    chunks = load_chunks(KNOWLEDGE_DIR)
    assert {c.metadata["source"] for c in chunks} == set(PUBLIC_DOCUMENTS)
    for chunk in chunks:
        assert chunk.metadata["doc_id"] and chunk.metadata["chunk_id"]
        assert chunk.page_content.strip()


def test_files_outside_allowlist_are_not_indexed(tmp_path):
    (tmp_path / "public.md").write_text(
        "---\nid: public\ntitle: Public\n---\n## A\nOpen info.\n"
    )
    (tmp_path / "bookings-export.md").write_text(
        "---\nid: x\ntitle: X\n---\n## A\nJohn Smith AB123CD\n"
    )
    chunks = load_chunks(tmp_path, allowlist=("public.md",))
    assert [c.metadata["doc_id"] for c in chunks] == ["public"]


def test_document_with_sensitive_data_is_refused(tmp_path):
    (tmp_path / "leak.md").write_text(
        "---\nid: leak\ntitle: Leak\n---\n## Contact\nWrite to boss@example.com\n"
    )
    with pytest.raises(KnowledgeError, match="sensitive"):
        load_chunks(tmp_path, allowlist=("leak.md",))


def test_front_matter_is_required_and_sections_split():
    with pytest.raises(KnowledgeError):
        parse_document("no front matter")
    assert split_sections("intro\n## A\nx\n## B\ny") == [
        ("", "intro"),
        ("A", "x"),
        ("B", "y"),
    ]
