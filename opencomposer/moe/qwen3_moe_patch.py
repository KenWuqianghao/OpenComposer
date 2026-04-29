"""Monkey-patch Qwen2/Qwen3 MoE blocks to support Composer-style router capture/replay."""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn.functional as F

from opencomposer.moe.router_replay import RouterReplayController

_ORIG_FORWARDS: dict[int, Any] = {}


def _patch_qwen3_sparse_moe_forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (  # type: ignore
        Qwen3MoeSparseMoeBlock,
    )

    assert isinstance(self, Qwen3MoeSparseMoeBlock)
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_flat = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states_flat)
    router_probs = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(router_probs, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states_flat.dtype)

    mode = RouterReplayController.mode()
    layer_idx = getattr(self, "_oc_layer_idx", -1)
    if mode == "capture":
        b = RouterReplayController.builder()
        if b is not None and layer_idx >= 0:
            b.set_layer(layer_idx, selected_experts)
    elif mode == "replay" and layer_idx >= 0:
        trace = RouterReplayController.replay_trace()
        if trace is not None:
            rep = trace.tensor_for_layer(layer_idx)
            if rep is not None:
                rep = rep.to(device=selected_experts.device, dtype=selected_experts.dtype)
                if rep.shape == selected_experts.shape:
                    selected_experts, routing_weights = RouterReplayController.merge_with_plausibility(
                        router_probs,
                        selected_experts,
                        rep,
                        tau=RouterReplayController.tau(),
                        norm_topk=self.norm_topk_prob,
                    )
                    routing_weights = routing_weights.to(hidden_states_flat.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_row in expert_hit:
        ei = int(expert_row.squeeze().item())
        expert_layer = self.experts[ei]
        idx, top_x = torch.where(expert_mask[ei].squeeze(0))
        current_state = hidden_states_flat[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states_flat.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def _patch_qwen2_sparse_moe_forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock  # type: ignore

    assert isinstance(self, Qwen2MoeSparseMoeBlock)
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_flat = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states_flat)
    router_probs = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(router_probs, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states_flat.dtype)

    mode = RouterReplayController.mode()
    layer_idx = getattr(self, "_oc_layer_idx", -1)
    if mode == "capture":
        b = RouterReplayController.builder()
        if b is not None and layer_idx >= 0:
            b.set_layer(layer_idx, selected_experts)
    elif mode == "replay" and layer_idx >= 0:
        trace = RouterReplayController.replay_trace()
        if trace is not None:
            rep = trace.tensor_for_layer(layer_idx)
            if rep is not None:
                rep = rep.to(device=selected_experts.device, dtype=selected_experts.dtype)
                if rep.shape == selected_experts.shape:
                    selected_experts, routing_weights = RouterReplayController.merge_with_plausibility(
                        router_probs,
                        selected_experts,
                        rep,
                        tau=RouterReplayController.tau(),
                        norm_topk=self.norm_topk_prob,
                    )
                    routing_weights = routing_weights.to(hidden_states_flat.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_row in expert_hit:
        ei = int(expert_row.squeeze().item())
        expert_layer = self.experts[ei]
        idx, top_x = torch.where(expert_mask[ei].squeeze(0))
        current_state = hidden_states_flat[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states_flat.dtype))

    shared_expert_output = self.shared_expert(hidden_states_flat)
    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_flat)) * shared_expert_output
    final_hidden_states = final_hidden_states + shared_expert_output

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def apply_moe_router_hooks(model: torch.nn.Module) -> None:
    """Patch MoE sparse blocks on an instantiated HF causal LM."""
    layer_idx = 0
    for module in model.modules():
        cls_name = module.__class__.__name__
        if cls_name == "Qwen3MoeSparseMoeBlock":
            mid = id(module)
            if mid not in _ORIG_FORWARDS:
                _ORIG_FORWARDS[mid] = module.forward
            module._oc_layer_idx = layer_idx  # type: ignore[attr-defined]
            module.forward = types.MethodType(_patch_qwen3_sparse_moe_forward, module)
            layer_idx += 1
        elif cls_name == "Qwen2MoeSparseMoeBlock":
            mid = id(module)
            if mid not in _ORIG_FORWARDS:
                _ORIG_FORWARDS[mid] = module.forward
            module._oc_layer_idx = layer_idx  # type: ignore[attr-defined]
            module.forward = types.MethodType(_patch_qwen2_sparse_moe_forward, module)
            layer_idx += 1


def remove_moe_router_hooks(model: torch.nn.Module) -> None:
    """Restore original forwards."""
    for module in model.modules():
        mid = id(module)
        if mid in _ORIG_FORWARDS:
            module.forward = _ORIG_FORWARDS[mid]  # type: ignore[assignment]
            del _ORIG_FORWARDS[mid]
            if hasattr(module, "_oc_layer_idx"):
                delattr(module, "_oc_layer_idx")


def count_moe_layers(model: torch.nn.Module) -> int:
    n = 0
    for module in model.modules():
        if module.__class__.__name__ in ("Qwen3MoeSparseMoeBlock", "Qwen2MoeSparseMoeBlock"):
            n += 1
    return n
