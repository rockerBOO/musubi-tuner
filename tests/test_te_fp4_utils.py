"""Tests for src/musubi_tuner/modules/te_fp4_utils.py."""

import contextlib

import pytest
import torch
import torch.nn as nn
import torch.utils.checkpoint

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
te = pytest.importorskip("transformer_engine.pytorch", reason="transformer_engine not installed")

from musubi_tuner.modules.te_fp4_utils import fp4_autocast, fp4_checkpoint_context_fn, swap_linears_to_te


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_TinyLinearHolder()])
        self.head = nn.Linear(8, 8)  # not under "blocks." -> must NOT be swapped


class _TinyLinearHolder(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(16, 16, bias=True)
        self.norm = nn.Linear(16, 16, bias=False)  # excluded by "norm"


@requires_cuda
def test_swap_linears_to_te_only_touches_targeted_linears():
    model = _TinyBlock().to("cuda", dtype=torch.bfloat16)
    orig_qkv_weight = model.blocks[0].qkv.weight.detach().clone()
    orig_qkv_bias = model.blocks[0].qkv.bias.detach().clone()

    swap_linears_to_te(model, target_keys=["blocks."], exclude_keys=["norm"])

    assert model.blocks[0].qkv.__class__.__name__ == "Linear"
    assert type(model.blocks[0].qkv).__module__.startswith("transformer_engine")
    assert torch.equal(model.blocks[0].qkv.weight.detach(), orig_qkv_weight)
    assert torch.equal(model.blocks[0].qkv.bias.detach(), orig_qkv_bias)

    # excluded by "norm" -> stays a plain nn.Linear
    assert type(model.blocks[0].norm).__module__ == "torch.nn.modules.linear"
    # outside "blocks." scope -> stays a plain nn.Linear
    assert type(model.head).__module__ == "torch.nn.modules.linear"


@requires_cuda
def test_swap_linears_to_te_preserves_forward_shape():
    model = _TinyBlock().to("cuda", dtype=torch.bfloat16)
    swap_linears_to_te(model, target_keys=["blocks."], exclude_keys=["norm"])
    x = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
    out = model.blocks[0].qkv(x)
    assert out.shape == (4, 16)


def test_fp4_autocast_disabled_is_nullcontext():
    ctx = fp4_autocast(False)
    assert isinstance(ctx, contextlib.nullcontext)


@requires_cuda
def test_fp4_autocast_enabled_engages_fp8_state():
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    with fp4_autocast(True):
        assert FP8GlobalStateManager.is_fp8_enabled() is True
    assert FP8GlobalStateManager.is_fp8_enabled() is False


def test_fp4_checkpoint_context_fn_disabled_returns_two_nullcontexts():
    fwd_ctx, recompute_ctx = fp4_checkpoint_context_fn(False)
    assert isinstance(fwd_ctx, contextlib.nullcontext)
    assert isinstance(recompute_ctx, contextlib.nullcontext)


def test_fp4_checkpoint_context_fn_forward_slot_is_always_nullcontext():
    # The forward pass already runs under the caller's ambient fp4_autocast context
    # (see call_dit); only the recompute slot needs to reactivate it.
    fwd_ctx, _ = fp4_checkpoint_context_fn(True)
    assert isinstance(fwd_ctx, contextlib.nullcontext)


@requires_cuda
def test_fp4_checkpoint_context_fn_enabled_recompute_slot_engages_fp8_state():
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    _, recompute_ctx = fp4_checkpoint_context_fn(True)
    with recompute_ctx:
        assert FP8GlobalStateManager.is_fp8_enabled() is True
    assert FP8GlobalStateManager.is_fp8_enabled() is False


@requires_cuda
def test_gradient_checkpointing_with_fp4_te_survives_recompute():
    """Reproduces the exact bug this fix addresses: without context_fn reactivating
    FP4 autocast during recompute, torch.utils.checkpoint.CheckpointError is raised
    because the recompute pass saves a different number of tensors than the original
    FP4-autocasted forward."""
    class _Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(128, 128, bias=True)

    model = _Holder().to("cuda", dtype=torch.bfloat16)
    swap_linears_to_te(model, target_keys=[""], exclude_keys=[])

    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def fn(x):
        return model.qkv(x)

    # Mirrors call_dit: fp4_autocast wraps only the initial checkpoint call, exits before
    # backward() runs. context_fn is the ONLY thing that can reactivate it for recompute --
    # without a working context_fn, this raises torch.utils.checkpoint.CheckpointError.
    with fp4_autocast(True):
        out = torch.utils.checkpoint.checkpoint(
            fn, x, use_reentrant=False, context_fn=lambda: fp4_checkpoint_context_fn(True)
        )
    out.sum().backward()
    assert x.grad is not None
