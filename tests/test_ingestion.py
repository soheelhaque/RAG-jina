"""Offline checks for whole-document loading."""

from pathlib import Path

import pytest

from src.ingestion import load_documents


def test_load_documents_preserves_text_and_source_in_filename_order(tmp_path: Path) -> None:
    text = "Definition: café earnings.\n\nLater reference to that definition.\n"
    (tmp_path / "02_note.txt").write_text(text, encoding="utf-8")
    (tmp_path / "01_note.txt").write_text("First document", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("Ignore me", encoding="utf-8")
    (tmp_path / "directory.txt").mkdir()

    documents = load_documents(tmp_path)

    assert [document.source for document in documents] == ["01_note.txt", "02_note.txt"]
    assert [document.text for document in documents] == ["First document", text]


def test_load_documents_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Document directory does not exist"):
        load_documents(tmp_path / "missing")


def test_load_documents_rejects_directory_without_text_files(tmp_path: Path) -> None:
    (tmp_path / "directory.txt").mkdir()
    with pytest.raises(ValueError, match="No .txt documents found"):
        load_documents(tmp_path)


def test_load_documents_loads_the_financial_corpus() -> None:
    directory = Path(__file__).resolve().parents[1] / "data" / "financial_docs"

    documents = load_documents(directory)

    assert len(documents) == 8
    assert all(document.text.strip() for document in documents)
    for document in documents:
        assert document.text == (directory / document.source).read_text(encoding="utf-8")
