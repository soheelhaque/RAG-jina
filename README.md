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

- `src/ingestion.py`: whole-document loading.
- `src/embeddings.py`: model loading and contextual token embeddings.
- `src/retrieval.py`: dense retrieval (to implement).
- `src/prompts.py`: generation prompt (to implement).
- `src/pipeline.py`: pipeline entry point.
- `data/financial_docs/`: eight shared financial documents.
The shared specification is supplied in the chat and is not stored in this repository.

The default demo only loads documents. Model loading is an explicit separate command;
indexing and generation remain unimplemented.
Dependencies are installed for local token embeddings, FAISS retrieval, and direct LLM API calls.
Model-specific requirements will be checked when integrating the Jina reference implementation.


## Whole-document token embeddings

Run from the project root:

```sh
uv run python -m scripts.inspect_embeddings
```

This loads `jinaai/jina-embeddings-v2-base-en` on CPU and embeds the first financial
document. The first run downloads model weights and Jina's custom model code
(`trust_remote_code=True`) into the ignored project-local `.cache/huggingface`.
`TokenEmbedder.load` accepts a `revision` argument for a reviewed model commit SHA;
its current default is `main`, which is not a reproducible pin.

The tokenizer processes the complete text without truncation. Documents exceeding
8,192 tokens, including special tokens, raise an error before model inference.
The final transformer layer produces a matrix of `[tokens, embedding dimensions]`:
each row represents one token in its document context. Character offsets and a
special-token mask stay aligned with those rows. Special tokens remain in the
matrix and must be excluded when we implement chunk pooling later.

The model runs in evaluation mode with gradient tracking disabled. There is no
chunking or pooling yet: these are token vectors, not vectors ready for a chunk index.

Validation: offline tests use a tiny randomly initialized BERT and a local tokenizer
to check real transformer inference, contextual effects, character alignment and
length handling. They do not establish Jina model compatibility or retrieval quality.
Python 3.13.9, Torch 2.14.1 and Transformers 5.20.2 import successfully, but the real
Jina download/inference and current reference implementation could not be checked:
the environment's network proxy was unreachable, and Hugging Face is absent from
its configured allowlist. No model weights were downloaded. The loader remains
provisional until that real-model check succeeds.
