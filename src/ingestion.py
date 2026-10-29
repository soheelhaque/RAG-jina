"""Load complete documents before any tokenisation or chunking."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """A document's source filename and its unmodified text."""

    source: str
    text: str


def load_documents(directory: Path) -> list[Document]:
    """Read UTF-8 text files in filename order, preserving each document whole."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {directory}")

    paths = sorted(path for path in directory.glob("*.txt") if path.is_file())
    if not paths:
        raise ValueError(f"No .txt documents found in: {directory}")

    return [Document(source=path.name, text=path.read_text(encoding="utf-8")) for path in paths]
