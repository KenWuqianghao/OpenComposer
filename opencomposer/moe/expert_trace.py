"""Serializable expert routing traces for MoE router replay."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class ExpertTrace:
    """Per-layer expert indices chosen at each token position (flattened batch*seq)."""

    layers: dict[int, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": self.meta, "layers": {}}
        for k, v in self.layers.items():
            if isinstance(v, torch.Tensor):
                payload["layers"][str(k)] = v.cpu().tolist()
            else:
                payload["layers"][str(k)] = v
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> ExpertTrace:
        raw = json.loads(Path(path).read_text())
        layers = {int(k): torch.tensor(v, dtype=torch.long) for k, v in raw["layers"].items()}
        return cls(layers=layers, meta=raw.get("meta") or {})

    def tensor_for_layer(self, layer_idx: int) -> torch.Tensor | None:
        t = self.layers.get(layer_idx)
        if t is None:
            return None
        return t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.long)


class ExpertTraceBuilder:
    """Accumulate routing decisions for one forward pass (one tensor per layer)."""

    def __init__(self) -> None:
        self._layers: dict[int, torch.Tensor] = {}
        self.meta: dict[str, Any] = {}

    def set_layer(self, layer_idx: int, selected_experts: torch.Tensor) -> None:
        """selected_experts shape (batch * seq, top_k)."""
        self._layers[layer_idx] = selected_experts.detach().cpu()

    def build(self) -> ExpertTrace:
        return ExpertTrace(layers=dict(self._layers), meta=dict(self.meta))
