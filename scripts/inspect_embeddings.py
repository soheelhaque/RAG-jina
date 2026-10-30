"""Inspect contextual token vectors for the first financial document."""

from pathlib import Path

from src.embeddings import MODEL_ID, TokenEmbedder
from src.ingestion import load_documents


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    document = load_documents(root / "data" / "financial_docs")[0]
    print(f"Loading {MODEL_ID}; the first run downloads model weights and custom code.")
    embedder = TokenEmbedder.load(cache_dir=root / ".cache" / "huggingface")
    result = embedder.embed(document)
    print(f"Source: {document.source}")
    print(f"Token embedding matrix: {tuple(result.vectors.shape)}")
    print("Rows include special tokens; each row is a contextual token vector.")
    for token_id, offset, special in zip(
        result.token_ids[:8], result.offsets[:8], result.special_tokens[:8], strict=True
    ):
        print(f"  id={token_id}, characters={offset}, special={special}")


if __name__ == "__main__":
    main()
