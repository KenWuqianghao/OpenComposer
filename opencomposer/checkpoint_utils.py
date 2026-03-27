"""Hugging Face Trainer checkpoint layout helpers."""

from __future__ import annotations

from pathlib import Path

from transformers import TrainerCallback


def prune_stale_monolithic_safetensors(output_dir: str) -> bool:
    """Remove ``model.safetensors`` when sharded weights + index are present.

    For ChatGLM (remote code), Trainer can leave a monolithic file whose tensors
    use generic ``model.*`` names while the sharded checkpoints use
    ``transformer.*``. Loading then picks the wrong file and trained weights are
    skipped.

    Returns:
        True if a stale file was removed.
    """
    out = Path(output_dir)
    index = out / "model.safetensors.index.json"
    monolithic = out / "model.safetensors"
    if index.is_file() and monolithic.is_file():
        monolithic.unlink()
        return True
    return False


def prune_stale_monolithic_under(output_dir: str) -> int:
    """Prune stale monolithic files in ``output_dir`` and ``checkpoint-*`` kids.

    Returns:
        Number of directories where a stale file was removed.
    """
    root = Path(output_dir)
    if not root.is_dir():
        return 0
    removed = 0
    candidates = [root] + sorted(root.glob("checkpoint-*"))
    for d in candidates:
        if d.is_dir() and prune_stale_monolithic_safetensors(str(d)):
            removed += 1
    return removed


def maybe_prune_before_hf_load(model_path: str, logger=None) -> None:
    """If ``model_path`` is a local folder, remove conflicting monolithic weights."""
    p = Path(model_path)
    if not p.is_dir():
        return
    n = prune_stale_monolithic_under(str(p.resolve()))
    if n and logger is not None:
        logger.info(
            "Removed stale model.safetensors in %d folder(s) under %s (prefer sharded ChatGLM weights).",
            n,
            model_path,
        )


class PruneStaleMonolithicCallback(TrainerCallback):
    """After each checkpoint write, drop monolithic ``model.safetensors`` if shards exist."""

    def on_save(self, args, state, control, **kwargs):
        if getattr(args, "local_rank", -1) > 0:
            return control
        prune_stale_monolithic_under(args.output_dir)
        return control
