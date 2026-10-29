"""Entry point for the incremental RAG implementation."""

from pathlib import Path

from src.ingestion import load_documents


def main() -> None:
    """Inspect the whole documents that will later be embedded."""
    directory = Path(__file__).resolve().parents[1] / "data" / "financial_docs"
    documents = load_documents(directory)
    print(f"Loaded {len(documents)} whole documents:")
    for document in documents:
        print(f"  {document.source}: {len(document.text):,} characters")
    print("Documents remain whole; tokenisation, embedding, and chunking come later.")


if __name__ == "__main__":
    main()
