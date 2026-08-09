"""Tests for NetworkTrainer.compute_loss's optional per-example reduction mode."""

import pytest
import torch

from musubi_tuner.training import parser_common
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer


def _args(**overrides):
    parser = parser_common.setup_parser_common()
    args, _ = parser.parse_known_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_reduction_none_returns_per_example_shape():
    trainer = NetworkTrainer()
    args = _args(weighting_scheme="none")
    # batch of 2, each example (1, 1, 2) elements
    pred = torch.tensor([[[[2.0, 0.0]]], [[[0.0, 3.0]]]])
    target = torch.zeros_like(pred)
    output = DiTOutput(pred=pred, target=target)
    timesteps = torch.tensor([500.0, 500.0])

    per_example, metrics = trainer.compute_loss(
        args, output, timesteps, None, torch.float32, torch.float32, 0, reduction="none"
    )

    assert per_example.shape == (2,)
    assert per_example[0].item() == pytest.approx((2.0**2 + 0.0**2) / 2)
    assert per_example[1].item() == pytest.approx((0.0**2 + 3.0**2) / 2)
    assert metrics == {}


def test_reduction_mean_matches_batch_average_of_none():
    trainer = NetworkTrainer()
    args = _args(weighting_scheme="none")
    pred = torch.tensor([[[[2.0, 0.0]]], [[[0.0, 3.0]]]])
    target = torch.zeros_like(pred)
    output = DiTOutput(pred=pred, target=target)
    timesteps = torch.tensor([500.0, 500.0])

    scalar, _ = trainer.compute_loss(args, output, timesteps, None, torch.float32, torch.float32, 0)
    per_example, _ = trainer.compute_loss(
        args, output, timesteps, None, torch.float32, torch.float32, 0, reduction="none"
    )

    assert scalar.item() == pytest.approx(per_example.mean().item())


def test_reduction_defaults_to_mean_and_returns_scalar():
    trainer = NetworkTrainer()
    args = _args(weighting_scheme="none")
    pred = torch.ones(2, 1, 1, 2)
    target = torch.zeros_like(pred)
    output = DiTOutput(pred=pred, target=target)
    timesteps = torch.tensor([500.0, 500.0])

    loss, _ = trainer.compute_loss(args, output, timesteps, None, torch.float32, torch.float32, 0)

    assert loss.ndim == 0
    assert loss.item() == pytest.approx(1.0)
