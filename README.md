# RAG-jina

A small Late Chunking RAG learning project using Jina AI's reference approach.

## Setup

Requires Python 3.13.9 and uv.

```sh
uv sync
uv run python scripts/run_demo.py
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

uv manages the local `.venv`; manual activation is optional.

## Structure

- `src/ingestion.py`: document loading and late-chunk embedding (to implement).
- `src/retrieval.py`: dense retrieval (to implement).
- `src/prompts.py`: generation prompt (to implement).
- `src/pipeline.py`: pipeline entry point.
- `data/financial_docs/`: shared financial corpus (to add).
- `docs/SPEC.md`: shared learning-series specification.

This scaffold does not yet download model weights, index documents, or call an LLM.
Dependencies are installed for local token embeddings, FAISS retrieval, and direct LLM API calls.
Model-specific requirements will be checked when integrating the Jina reference implementation.
