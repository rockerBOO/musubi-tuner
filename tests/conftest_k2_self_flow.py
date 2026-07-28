"""Shared fixtures: tiny CPU-only Krea 2 (K2) model for self-flow tests."""

import pytest
import torch

from musubi_tuner.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT


@pytest.fixture
def tiny_k2_config():
    """Minimal K2 config for fast CPU tests — not real weights.

    features=32, heads=2 -> headdim=16 -> axes=[4,6,6] (sum=16, all even),
    the constraint SingleStreamDiT.__init__ asserts on. layers=3 gives room
    for student_feature_layer=0 / teacher_feature_layer=2 in tests.
    """
    return SingleMMDiTConfig(
        features=32,
        tdim=32,
        txtdim=32,
        heads=2,
        multiplier=1,
        layers=3,
        patch=2,
        channels=4,
        bias=False,
        theta=1e3,
        kvheads=None,
        txtlayers=1,
        txtheads=2,
        txtkvheads=2,
    )


@pytest.fixture
def tiny_k2_model(tiny_k2_config):
    torch.manual_seed(0)
    model = SingleStreamDiT(tiny_k2_config, attn_mode="torch")
    model.eval()
    return model


def make_k2_batch(B=2, H=8, W=8, n_txt=3, patch=2, txtlayers=1, txtdim=32, seed=0):
    """Build minimal K2 training inputs: (batch dict, latents, noise).

    latents/noise are (B, C, 1, H, W) — K2 forces single-frame 5D latents.
    krea2_vl_embed is a list of per-sample (seq_len, txtlayers, txtdim) tensors
    (variable seq_len across the batch, matching the real varlen text cache).
    """
    torch.manual_seed(seed)
    channels = 4
    latents = torch.randn(B, channels, 1, H, W)
    noise = torch.randn_like(latents)
    vl_embed = [torch.randn(n_txt, txtlayers, txtdim) for _ in range(B)]
    batch = {"krea2_vl_embed": vl_embed, "timesteps": None}
    return batch, latents, noise
