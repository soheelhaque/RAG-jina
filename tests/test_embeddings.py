"""Exercise token alignment and real transformer inference without downloads."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from src.embeddings import MODEL_ID, TokenEmbedder
from src.ingestion import Document


@pytest.fixture
def embedder() -> TokenEmbedder:
    backend = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "café": 3, "earnings": 4},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    model = BertModel(
        BertConfig(
            vocab_size=5,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=16,
        )
    )
    return TokenEmbedder(tokenizer, model, max_tokens=16)


def test_whole_document_offsets_and_inference(embedder: TokenEmbedder) -> None:
    document = Document("note.txt", "café\n\nearnings")
    result = embedder.embed(document)
    assert result.document is document
    assert result.vectors.shape == (4, 8)
    assert result.token_ids == [1, 3, 4, 2]
    assert result.offsets == [(0, 0), (0, 4), (6, 14), (0, 0)]
    assert result.special_tokens == [True, False, False, True]
    assert torch.isfinite(result.vectors).all()
    assert not result.vectors.requires_grad
    assert not embedder.model.training
    torch.testing.assert_close(result.vectors, embedder.embed(document).vectors)


def test_later_context_changes_earlier_token(embedder: TokenEmbedder) -> None:
    short = embedder.embed(Document("short", "café"))
    longer = embedder.embed(Document("longer", "café earnings"))
    assert not torch.equal(short.vectors[1], longer.vectors[1])


def test_length_limit_includes_special_tokens(embedder: TokenEmbedder) -> None:
    embedder.max_tokens = 4
    embedder.embed(Document("fits", "café earnings"))
    model = Mock()
    embedder.model = model
    with pytest.raises(ValueError, match="5 tokens exceeds"):
        embedder.embed(Document("too-long", "café earnings café"))
    model.assert_not_called()


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_reject_empty_text(embedder: TokenEmbedder, text: str) -> None:
    with pytest.raises(ValueError, match="Document is empty"):
        embedder.embed(Document("empty", text))


def test_reject_slow_tokenizer() -> None:
    with pytest.raises(ValueError, match="fast tokenizer"):
        TokenEmbedder(SimpleNamespace(is_fast=False), Mock())


def test_explicit_loader_uses_same_revision_and_cache(monkeypatch, tmp_path) -> None:
    tokenizer = Mock(is_fast=True)
    model = Mock()
    tokenizer_loader = Mock(return_value=tokenizer)
    model_loader = Mock(return_value=model)
    monkeypatch.setattr("src.embeddings.AutoTokenizer.from_pretrained", tokenizer_loader)
    monkeypatch.setattr("src.embeddings.AutoModel.from_pretrained", model_loader)
    TokenEmbedder.load(tmp_path, revision="reviewed-commit")
    tokenizer_loader.assert_called_once_with(
        MODEL_ID, revision="reviewed-commit", cache_dir=tmp_path, use_fast=True
    )
    model_loader.assert_called_once_with(
        MODEL_ID, revision="reviewed-commit", cache_dir=tmp_path, trust_remote_code=True
    )
