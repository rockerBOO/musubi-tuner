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
