"""Explorative Modeling (XM) — best-of-K training, architecture-generic mixin.

Reference: https://explorative-modeling.github.io/ (Forward XM).

For each training example, sample K noise candidates, score all K, and
backward only the lowest-loss candidate per example. No model edits required:
candidates are built and scored entirely through the existing
``NetworkTrainer`` extension seams (``call_dit``, ``compute_loss``).

Internal extension point — no API stability guarantees. Subclasses live in
this repo; if you fork, expect breakage on updates.
"""

import torch


def _select_winner(loss_stack: torch.Tensor) -> torch.Tensor:
    """Per-example argmin over K candidates.

    ``loss_stack``: ``(K, B)`` — per-candidate, per-example loss. Returns a
    ``(B,)`` long tensor: the winning candidate index for each example.
    """
    return loss_stack.argmin(dim=0)


def _gather_winner(stack: torch.Tensor, winner: torch.Tensor) -> torch.Tensor:
    """Select each example's winning candidate out of a stacked tensor.

    ``stack``: ``(K, B, ...)`` — candidate tensors sharing a per-example
    layout (noise, noisy input, pred, target, ...). ``winner``: ``(B,)``,
    as returned by ``_select_winner``. Returns ``(B, ...)``.
    """
    batch_idx = torch.arange(stack.shape[1], device=stack.device)
    return stack[winner, batch_idx]
