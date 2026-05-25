import pytest
from musubi_tuner.training.trainer_base import NetworkTrainer


def test_on_before_backward_exists_and_is_callable():
    """on_before_backward is a no-op on the base class."""
    trainer = NetworkTrainer.__new__(NetworkTrainer)
    import torch
    loss = torch.tensor(1.0)
    result = trainer.on_before_backward(loss)
    assert result is None


def test_on_after_backward_exists_and_is_callable():
    """on_after_backward is a no-op on the base class."""
    trainer = NetworkTrainer.__new__(NetworkTrainer)
    result = trainer.on_after_backward()
    assert result is None


def test_hooks_are_overrideable():
    """Subclass can override hooks and they get called."""
    called = []

    class PatchedTrainer(NetworkTrainer):
        def on_before_backward(self, loss):
            called.append(("before", loss))

        def on_after_backward(self):
            called.append("after")

    trainer = PatchedTrainer.__new__(PatchedTrainer)
    import torch
    loss = torch.tensor(2.5)
    trainer.on_before_backward(loss)
    trainer.on_after_backward()

    assert called[0][0] == "before"
    assert called[1] == "after"
