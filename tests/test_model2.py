from __future__ import annotations

from room_matcher.model2 import DEFAULT_HF_MODEL_NAME, resolve_tokenizer_name


def test_resolve_tokenizer_name_uses_override_for_multilingual_minilm() -> None:
    assert resolve_tokenizer_name(DEFAULT_HF_MODEL_NAME) == "xlm-roberta-base"


def test_resolve_tokenizer_name_respects_explicit_value() -> None:
    assert resolve_tokenizer_name(DEFAULT_HF_MODEL_NAME, "custom-tokenizer") == "custom-tokenizer"
