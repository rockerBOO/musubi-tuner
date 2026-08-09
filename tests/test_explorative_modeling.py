"""Tests for Explorative Modeling (XM): best-of-K training.

Reference: https://explorative-modeling.github.io/
"""

import torch

from musubi_tuner.training.explorative_modeling import _gather_winner, _select_winner


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
