"""Obtain contextual token vectors before choosing any chunk boundaries."""

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.ingestion import Document

MODEL_ID = "jinaai/jina-embeddings-v2-base-en"
MAX_TOKENS = 8192


@dataclass(frozen=True)
class TokenEmbeddings:
    """Aligned rows, token IDs and character spans, including special tokens.

    Special tokens have a true mask entry and usually the offset (0, 0).
    Offsets are half-open character spans in the original document text.
    """

    document: Document
    vectors: torch.Tensor
    token_ids: list[int]
    offsets: list[tuple[int, int]]
    special_tokens: list[bool]


class TokenEmbedder:
    """Run one whole document through a transformer on CPU, without pooling."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required for character offsets.")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        self.tokenizer = tokenizer
        self.model = model.to("cpu").eval()
        self.max_tokens = max_tokens

    @classmethod
    def load(cls, cache_dir: Path, revision: str = "main") -> "TokenEmbedder":
        """Download/load Jina's tokenizer and custom model code explicitly.

        Jina v2 requires trust_remote_code. A reviewed commit SHA can be passed
        as revision to pin both code and weights to the same model revision.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=revision, cache_dir=cache_dir, use_fast=True
        )
        model = AutoModel.from_pretrained(
            MODEL_ID, revision=revision, cache_dir=cache_dir, trust_remote_code=True
        )
        return cls(tokenizer, model)

    def embed(self, document: Document) -> TokenEmbeddings:
        """Return final-layer token vectors; never silently truncate the text."""
        if not document.text.strip():
            raise ValueError(f"Document is empty: {document.source}")
        encoded = self.tokenizer(
            document.text,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            truncation=False,
            padding=False,
        )
        offsets = [tuple(span) for span in encoded.pop("offset_mapping")[0].tolist()]
        special_tokens = encoded.pop("special_tokens_mask")[0].bool().tolist()
        token_ids = encoded["input_ids"][0].tolist()
        if len(token_ids) > self.max_tokens:
            raise ValueError(
                f"{document.source}: {len(token_ids)} tokens exceeds the "
                f"{self.max_tokens}-token limit (including special tokens)."
            )
        with torch.inference_mode():
            vectors = self.model(**encoded).last_hidden_state[0].detach().cpu()
        if vectors.ndim != 2 or vectors.shape[0] != len(token_ids):
            raise ValueError("Model output must contain one vector per input token.")
        return TokenEmbeddings(document, vectors, token_ids, offsets, special_tokens)
