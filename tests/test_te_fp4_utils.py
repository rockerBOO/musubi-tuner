"""Tests for src/musubi_tuner/modules/te_fp4_utils.py."""

import contextlib

import pytest
import torch
import torch.nn as nn

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
te = pytest.importorskip("transformer_engine.pytorch", reason="transformer_engine not installed")

from musubi_tuner.modules.te_fp4_utils import fp4_autocast, swap_linears_to_te


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
