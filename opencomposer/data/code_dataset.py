"""Code corpus loading and preprocessing for continued pretraining (Stage 1)."""

from __future__ import annotations

import logging
from typing import Iterator

from datasets import IterableDataset, load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ["python", "javascript", "typescript", "go", "rust"]

# Approximate token-to-byte ratio for code
_BYTES_PER_TOKEN_ESTIMATE = 3.5


def load_code_corpus(
    dataset_name: str = "bigcode/starcoderdata",
    languages: list[str] | None = None,
    streaming: bool = True,
    split: str = "train",
) -> IterableDataset:
    """Load a code corpus from HuggingFace, optionally filtering by language."""
    languages = languages or SUPPORTED_LANGUAGES
    logger.info("Loading code corpus: %s (languages=%s, streaming=%s)", dataset_name, languages, streaming)

    datasets_by_lang = []
    for lang in languages:
        try:
            ds = load_dataset(dataset_name, data_dir=lang, split=split, streaming=streaming)
            datasets_by_lang.append(ds)
            logger.info("Loaded %s/%s", dataset_name, lang)
        except Exception:
            logger.warning("Could not load language %s from %s, skipping", lang, dataset_name)

    if not datasets_by_lang:
        logger.warning(
            "No languages loaded from %s (may need HF auth or dataset access).",
            dataset_name,
        )
        raise RuntimeError(f"No languages loaded from {dataset_name}")

    from datasets import interleave_datasets
    combined = interleave_datasets(datasets_by_lang)
    return combined


def load_fallback_code_stream(
    dataset_name: str = "wikitext",
    dataset_config: str | None = "wikitext-103-raw-v1",
    split: str = "train",
    streaming: bool = True,
) -> IterableDataset:
    """Load a public dataset when gated code corpora are unavailable (smoke / CPT fallback)."""
    logger.info(
        "Loading fallback corpus: %s config=%s (streaming=%s)",
        dataset_name,
        dataset_config,
        streaming,
    )
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)
    return load_dataset(dataset_name, split=split, streaming=streaming)


def tokenize_and_chunk(
    dataset: IterableDataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 4096,
    text_column: str = "content",
) -> IterableDataset:
    """Tokenize a streaming dataset and pack into fixed-length chunks."""

    def _generator() -> Iterator[dict]:
        buffer_ids: list[int] = []
        for example in dataset:
            text = example.get(text_column, "")
            if not text or len(text) < 50:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            buffer_ids.extend(ids)
            buffer_ids.append(tokenizer.eos_token_id)

            while len(buffer_ids) >= max_seq_length:
                chunk = buffer_ids[:max_seq_length]
                buffer_ids = buffer_ids[max_seq_length:]
                yield {
                    "input_ids": chunk,
                    "labels": chunk,
                    "attention_mask": [1] * max_seq_length,
                }

    return IterableDataset.from_generator(_generator)


def create_code_pretraining_dataset(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str = "bigcode/starcoderdata",
    languages: list[str] | None = None,
    max_seq_length: int = 4096,
    streaming: bool = True,
    fallback_dataset: str | None = None,
    fallback_dataset_config: str | None = None,
    fallback_text_column: str = "text",
) -> IterableDataset:
    """Full pipeline: load code corpus -> tokenize -> chunk."""
    try:
        raw = load_code_corpus(dataset_name=dataset_name, languages=languages, streaming=streaming)
        text_column = "content"
    except RuntimeError:
        if not fallback_dataset:
            raise
        logger.info("Using fallback dataset %s (column=%s)", fallback_dataset, fallback_text_column)
        raw = load_fallback_code_stream(
            dataset_name=fallback_dataset,
            dataset_config=fallback_dataset_config,
            streaming=streaming,
        )
        text_column = fallback_text_column

    return tokenize_and_chunk(
        raw, tokenizer, max_seq_length=max_seq_length, text_column=text_column
    )
