"""Tests for the rank_pattern network_arg: per-module rank/alpha override by regex."""

import logging

import pytest
import torch.nn as nn

from musubi_tuner.networks.lora import LoRANetwork, create_network


def _make_network(rank_pattern=None, **kwargs):
    """Minimal LoRANetwork with no target_replace_modules, so no modules are actually
    wrapped -- just enough to exercise rank_pattern parsing/compilation in isolation."""
    return LoRANetwork(
        target_replace_modules=[],
        prefix="lora_unet",
        text_encoders=[],
        unet=nn.Module(),
        lora_dim=8,
        alpha=8,
        rank_pattern=rank_pattern,
        **kwargs,
    )


def test_rank_pattern_none_gives_empty_list():
    network = _make_network(rank_pattern=None)
    assert network.rank_patterns == []


def test_rank_pattern_compiles_regex_dim_alpha():
    network = _make_network(rank_pattern=["^blocks\\.0\\.attn\\.wk$:2:2"])
    assert len(network.rank_patterns) == 1
    compiled, dim, alpha = network.rank_patterns[0]
    assert compiled.fullmatch("blocks.0.attn.wk")
    assert not compiled.fullmatch("blocks.1.attn.wk")
    assert dim == 2
    assert alpha == 2.0


def test_rank_pattern_empty_alpha_field_is_none():
    network = _make_network(rank_pattern=["^blocks\\.0\\.attn\\.wq$:5:"])
    _, dim, alpha = network.rank_patterns[0]
    assert dim == 5
    assert alpha is None


def test_rank_pattern_preserves_list_order():
    network = _make_network(
        rank_pattern=[
            "^blocks\\.0\\.attn\\.wv$:2:2",
            "^blocks\\.\\d+\\.attn\\.wv$:6:6",
        ]
    )
    dims = [dim for _, dim, _ in network.rank_patterns]
    assert dims == [2, 6]


def test_rank_pattern_colon_inside_regex_non_capturing_group():
    network = _make_network(rank_pattern=["^blocks\\.0\\.(?:attn)\\.wv$:3:3"])
    compiled, dim, alpha = network.rank_patterns[0]
    assert compiled.fullmatch("blocks.0.attn.wv")
    assert dim == 3
    assert alpha == 3.0


def test_rank_pattern_invalid_regex_is_skipped_not_raised(caplog):
    with caplog.at_level(logging.ERROR):
        network = _make_network(
            rank_pattern=["(unclosed:4:4", "^blocks\\.0\\.attn\\.wq$:2:2"]
        )
    assert len(network.rank_patterns) == 1
    _, dim, alpha = network.rank_patterns[0]
    assert dim == 2
    assert alpha == 2.0
    assert "Invalid rank_pattern regex" in caplog.text


def test_create_network_literal_evals_rank_pattern_string():
    network = create_network(
        target_replace_modules=[],
        prefix="lora_unet",
        multiplier=1.0,
        network_dim=8,
        network_alpha=8,
        vae=None,
        text_encoders=[],
        unet=nn.Module(),
        rank_pattern="['^blocks\\\\.0\\\\.attn\\\\.wk$:2:2']",
    )
    assert len(network.rank_patterns) == 1
    compiled, dim, alpha = network.rank_patterns[0]
    assert compiled.fullmatch("blocks.0.attn.wk")
    assert dim == 2
    assert alpha == 2.0


HUNYUAN_DOUBLE = "MMDoubleStreamBlock"


class _FakeAttn(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)


class MMDoubleStreamBlock(nn.Module):
    """Named to match a target_replace_modules entry, like the real block classes."""

    def __init__(self, dim=8):
        super().__init__()
        self.attn = _FakeAttn(dim)


class _FakeUnet(nn.Module):
    def __init__(self, n_blocks=2, dim=8):
        super().__init__()
        self.blocks = nn.ModuleList([MMDoubleStreamBlock(dim) for _ in range(n_blocks)])


def _lora_dims_by_name(network):
    return {lora.lora_name: (lora.lora_dim, float(lora.alpha)) for lora in network.unet_loras}


def test_rank_pattern_overrides_matched_module_only():
    network = create_network(
        target_replace_modules=[HUNYUAN_DOUBLE],
        prefix="lora_unet",
        multiplier=1.0,
        network_dim=8,
        network_alpha=8,
        vae=None,
        text_encoders=[],
        unet=_FakeUnet(n_blocks=2),
        rank_pattern="['^blocks\\.0\\.attn\\.wk$:2:2']",
    )
    dims = _lora_dims_by_name(network)
    assert dims["lora_unet_blocks_0_attn_wk"] == (2, 2.0)
    # everything else keeps the network-wide default
    assert dims["lora_unet_blocks_0_attn_wq"] == (8, 8.0)
    assert dims["lora_unet_blocks_1_attn_wk"] == (8, 8.0)


def test_rank_pattern_first_match_wins():
    network = create_network(
        target_replace_modules=[HUNYUAN_DOUBLE],
        prefix="lora_unet",
        multiplier=1.0,
        network_dim=8,
        network_alpha=8,
        vae=None,
        text_encoders=[],
        unet=_FakeUnet(n_blocks=1),
        rank_pattern=(
            "['^blocks\\.0\\.attn\\.wv$:2:2', "
            "'^blocks\\.\\d+\\.attn\\.wv$:6:6']"
        ),
    )
    dims = _lora_dims_by_name(network)
    assert dims["lora_unet_blocks_0_attn_wv"] == (2, 2.0)


def test_rank_pattern_empty_alpha_falls_back_to_network_alpha():
    network = create_network(
        target_replace_modules=[HUNYUAN_DOUBLE],
        prefix="lora_unet",
        multiplier=1.0,
        network_dim=8,
        network_alpha=8,
        vae=None,
        text_encoders=[],
        unet=_FakeUnet(n_blocks=1),
        rank_pattern="['^blocks\\.0\\.attn\\.wq$:5:']",
    )
    dims = _lora_dims_by_name(network)
    assert dims["lora_unet_blocks_0_attn_wq"] == (5, 8.0)


def test_rank_pattern_dim_zero_skips_module():
    network = create_network(
        target_replace_modules=[HUNYUAN_DOUBLE],
        prefix="lora_unet",
        multiplier=1.0,
        network_dim=8,
        network_alpha=8,
        vae=None,
        text_encoders=[],
        unet=_FakeUnet(n_blocks=1),
        rank_pattern="['^blocks\\.0\\.attn\\.wv$:0:']",
    )
    dims = _lora_dims_by_name(network)
    assert "lora_unet_blocks_0_attn_wv" not in dims
    assert "lora_unet_blocks_0_attn_wq" in dims


def test_rank_pattern_matches_regex_with_non_capturing_group():
    network = create_network(
        target_replace_modules=[HUNYUAN_DOUBLE],
        prefix="lora_unet",
        multiplier=1.0,
        network_dim=8,
        network_alpha=8,
        vae=None,
        text_encoders=[],
        unet=_FakeUnet(n_blocks=1),
        rank_pattern="['^blocks\\.0\\.(?:attn)\\.wv$:3:3']",
    )
    dims = _lora_dims_by_name(network)
    assert dims["lora_unet_blocks_0_attn_wv"] == (3, 3.0)
