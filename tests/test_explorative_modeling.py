"""Tests for Explorative Modeling (XM): best-of-K training.

Reference: https://explorative-modeling.github.io/
"""

import pytest
import torch

from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.training import parser_common
from musubi_tuner.training.explorative_modeling import (
    ExplorativeModelingMixin,
    _gather_winner,
    _select_winner,
    explorative_modeling_setup_parser,
)
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer


def test_select_winner_is_per_example_not_per_batch():
    # K=3 candidates, B=2 examples. Different winner per example.
    loss_stack = torch.tensor(
        [
            [4.0, 1.0],  # candidate 0
            [1.0, 4.0],  # candidate 1
            [9.0, 9.0],  # candidate 2
        ]
    )
    winner = _select_winner(loss_stack)
    assert winner.tolist() == [1, 0]


def test_select_winner_all_same_candidate():
    loss_stack = torch.tensor([[5.0, 5.0], [1.0, 1.0], [9.0, 9.0]])
    winner = _select_winner(loss_stack)
    assert winner.tolist() == [1, 1]


def test_gather_winner_selects_matching_candidate_1d():
    stack = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])  # (K=3, B=2)
    winner = torch.tensor([1, 0])
    gathered = _gather_winner(stack, winner)
    assert gathered.tolist() == [30.0, 20.0]


def test_gather_winner_multi_dim():
    stack = torch.arange(2 * 2 * 3).reshape(2, 2, 3).float()  # (K=2, B=2, C=3)
    winner = torch.tensor([1, 0])
    gathered = _gather_winner(stack, winner)
    assert gathered.shape == (2, 3)
    assert torch.equal(gathered[0], stack[1, 0])
    assert torch.equal(gathered[1], stack[0, 1])


def _xm_args(**overrides):
    parser = parser_common.setup_parser_common()
    parser = explorative_modeling_setup_parser(parser)
    args, _ = parser.parse_known_args([])
    args.timestep_sampling = "uniform"
    args.weighting_scheme = "none"
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class _FakeAccelerator:
    device = torch.device("cpu")


def _latents_and_noise(batch_size=2, shape=(4, 1, 2, 2)):
    latents = torch.zeros(batch_size, *shape)
    noise = torch.ones_like(latents)
    return latents, noise


class _ScriptedTrainer(ExplorativeModelingMixin, NetworkTrainer):
    """Test double: call_dit ignores its tensor inputs and returns a scripted
    per-example loss for each successive invocation, in call order. Lets us
    control exactly what each candidate's loss is without a real transformer.
    """

    def __init__(self, scripted_losses):
        super().__init__()
        self._scripted_losses = [torch.tensor(vals, dtype=torch.float32) for vals in scripted_losses]
        self.call_count = 0

    def call_dit(self, args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs):
        vals = self._scripted_losses[self.call_count]
        self.call_count += 1
        shape = latents.shape
        pred = torch.sqrt(vals).view(-1, *([1] * (len(shape) - 1))).expand(shape).clone()
        target = torch.zeros_like(pred)
        return DiTOutput(pred=pred, target=target)


def test_process_batch_disabled_falls_through_to_base():
    latents, noise = _latents_and_noise()
    noise_scheduler = FlowMatchDiscreteScheduler()
    args = _xm_args(explorative_modeling=False)
    batch = {"timesteps": [0.3, 0.7]}
    trainer = _ScriptedTrainer([[4.0, 1.0]])

    loss, metrics = trainer.process_batch(
        args, _FakeAccelerator(), None, None, batch, latents, noise, noise_scheduler,
        torch.float32, torch.float32, None, 0,
    )

    assert trainer.call_count == 1
    assert "xm/k" not in metrics


def test_process_batch_memory_efficient_scores_then_regenerates():
    latents, noise = _latents_and_noise()
    noise_scheduler = FlowMatchDiscreteScheduler()
    args = _xm_args(explorative_modeling=True, explorative_modeling_k=3, explorative_modeling_memory_efficient=True)
    batch = {"timesteps": [0.3, 0.7]}

    # K=3 scoring calls, per-example losses:
    #   ex0: [4, 1, 9] -> winner candidate 1 (loss 1)
    #   ex1: [1, 4, 9] -> winner candidate 0 (loss 1)
    scored = [[4.0, 1.0], [1.0, 4.0], [9.0, 9.0]]
    # 4th call: the memory-efficient regeneration forward on the gathered winners.
    regenerated = [1.0, 1.0]
    trainer = _ScriptedTrainer(scored + [regenerated])

    loss, metrics = trainer.process_batch(
        args, _FakeAccelerator(), None, None, batch, latents, noise, noise_scheduler,
        torch.float32, torch.float32, None, 0,
    )

    assert trainer.call_count == 4  # K scoring calls + 1 regeneration
    assert loss.item() == pytest.approx(1.0, rel=1e-4)
    assert metrics["xm/k"] == 3.0
    assert metrics["xm/candidate_loss_std"] > 0.0


def test_process_batch_literal_mode_reuses_scored_forward_no_extra_call():
    latents, noise = _latents_and_noise()
    noise_scheduler = FlowMatchDiscreteScheduler()
    args = _xm_args(explorative_modeling=True, explorative_modeling_k=3, explorative_modeling_memory_efficient=False)
    batch = {"timesteps": [0.3, 0.7]}

    # Same per-example winners as above: ex0 -> candidate 1 (loss 1), ex1 -> candidate 0 (loss 1).
    scored = [[4.0, 1.0], [1.0, 4.0], [9.0, 9.0]]
    trainer = _ScriptedTrainer(scored)  # no extra scripted loss for a regeneration call

    loss, metrics = trainer.process_batch(
        args, _FakeAccelerator(), None, None, batch, latents, noise, noise_scheduler,
        torch.float32, torch.float32, None, 0,
    )

    assert trainer.call_count == 3  # no extra forward — literal mode gathers from the scored outputs
    # Winner losses are 1.0 and 1.0 for the two examples -> mean 1.0. A wrong gather
    # (e.g. picking candidate 0 for both) would give (4.0 + 1.0) / 2 = 2.5 instead.
    assert loss.item() == pytest.approx(1.0, rel=1e-4)


def test_process_batch_k1_matches_disabled_numerically():
    latents, noise = _latents_and_noise()
    noise_scheduler = FlowMatchDiscreteScheduler()
    batch = {"timesteps": [0.3, 0.7]}
    scripted = [[4.0, 1.0]]

    args_off = _xm_args(explorative_modeling=False)
    args_on = _xm_args(explorative_modeling=True, explorative_modeling_k=1, explorative_modeling_memory_efficient=False)

    trainer_off = _ScriptedTrainer(list(scripted))
    trainer_on = _ScriptedTrainer(list(scripted))

    loss_off, _ = trainer_off.process_batch(
        args_off, _FakeAccelerator(), None, None, batch, latents, noise, noise_scheduler,
        torch.float32, torch.float32, None, 0,
    )
    loss_on, _ = trainer_on.process_batch(
        args_on, _FakeAccelerator(), None, None, batch, latents, noise, noise_scheduler,
        torch.float32, torch.float32, None, 0,
    )

    assert loss_on.item() == pytest.approx(loss_off.item(), rel=1e-4)


def test_extra_metadata_includes_xm_keys_when_enabled():
    trainer = _ScriptedTrainer([])
    args = _xm_args(explorative_modeling=True, explorative_modeling_k=5, explorative_modeling_memory_efficient=False)
    metadata = trainer.extra_metadata(args)
    assert metadata["ss_xm"] is True
    assert metadata["ss_xm_k"] == 5
    assert metadata["ss_xm_memory_efficient"] is False


def test_extra_metadata_empty_when_disabled():
    trainer = _ScriptedTrainer([])
    args = _xm_args(explorative_modeling=False)
    metadata = trainer.extra_metadata(args)
    assert metadata == {}


def test_parser_defaults():
    parser = explorative_modeling_setup_parser(parser_common.setup_parser_common())
    args, _ = parser.parse_known_args([])
    assert args.explorative_modeling is False
    assert args.explorative_modeling_k == 4
    assert args.explorative_modeling_memory_efficient is True


def test_parser_accepts_overrides():
    parser = explorative_modeling_setup_parser(parser_common.setup_parser_common())
    args, _ = parser.parse_known_args(
        ["--explorative_modeling", "--explorative_modeling_k", "8", "--no-explorative_modeling_memory_efficient"]
    )
    assert args.explorative_modeling is True
    assert args.explorative_modeling_k == 8
    assert args.explorative_modeling_memory_efficient is False
