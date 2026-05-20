"""Tests for TrainerOrchestrator dispatch machinery.

Stubs avoid importing torch/accelerate so tests run without GPU.
"""

import pytest
from musubi_tuner.training.orchestrator import TrainerOrchestrator
from musubi_tuner.training.trainer_base import NetworkTrainer


# --- Minimal stubs that don't need GPU/models ---

class _BaseExt(NetworkTrainer):
    """Extension that overrides nothing — should not appear in any dispatch table."""
    pass


class _VoidExt(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.calls = []

    def on_train_start(self, **kwargs):
        self.calls.append("on_train_start")

    def on_post_optimizer_step(self, **kwargs):
        self.calls.append("on_post_optimizer_step")


class _DictExt(NetworkTrainer):
    def extra_metadata(self, args):
        return {"key_a": "val_a"}

    def extra_step_logs(self, args, logs):
        return {"log_a": 1.0}


class _DictExt2(NetworkTrainer):
    def extra_metadata(self, args):
        return {"key_b": "val_b"}

    def extra_step_logs(self, args, logs):
        return {"log_b": 2.0}


class _ChainExt(NetworkTrainer):
    def extra_trainable_params(self, args, accelerator, network, transformer, trainable_params):
        return trainable_params + ["param_a"]


class _ChainExt2(NetworkTrainer):
    def extra_trainable_params(self, args, accelerator, network, transformer, trainable_params):
        return trainable_params + ["param_b"]


class _ProcessBatchExt(NetworkTrainer):
    def process_batch(self, **kwargs):
        return None, {}


class _ComputeLossExt(NetworkTrainer):
    def compute_loss(self, **kwargs):
        return None, {}


# Concrete orchestrator (no GPU-needing overrides)
class _TestOrchestrator(TrainerOrchestrator):
    pass


# --- Dispatch table tests ---

def test_add_extension_no_override_not_in_table():
    orch = _TestOrchestrator()
    orch.add_extension(_BaseExt())
    assert "on_train_start" not in orch._dispatch_table
    assert "extra_metadata" not in orch._dispatch_table


def test_add_extension_void_override_registered():
    orch = _TestOrchestrator()
    ext = _VoidExt()
    orch.add_extension(ext)
    assert ext in orch._dispatch_table["on_train_start"]
    assert ext in orch._dispatch_table["on_post_optimizer_step"]


def test_add_extension_dict_override_registered():
    orch = _TestOrchestrator()
    ext = _DictExt()
    orch.add_extension(ext)
    assert ext in orch._dispatch_table["extra_metadata"]
    assert ext in orch._dispatch_table["extra_step_logs"]


def test_add_extension_multiple_same_hook():
    orch = _TestOrchestrator()
    e1, e2 = _VoidExt(), _VoidExt()
    orch.add_extension(e1)
    orch.add_extension(e2)
    assert orch._dispatch_table["on_train_start"] == [e1, e2]


# --- Void hook tests ---

def test_void_hook_calls_all_in_order():
    orch = _TestOrchestrator()
    e1, e2 = _VoidExt(), _VoidExt()
    orch.add_extension(e1)
    orch.add_extension(e2)
    orch.on_train_start()
    assert e1.calls == ["on_train_start"]
    assert e2.calls == ["on_train_start"]


def test_void_hook_with_no_extensions_is_noop():
    orch = _TestOrchestrator()
    orch.on_train_start()  # should not raise


# --- Dict merge tests ---

def test_extra_metadata_merges_all():
    orch = _TestOrchestrator()
    orch.add_extension(_DictExt())
    orch.add_extension(_DictExt2())
    result = orch.extra_metadata(args=None)
    assert result == {"key_a": "val_a", "key_b": "val_b"}


def test_extra_metadata_empty_when_no_extensions():
    orch = _TestOrchestrator()
    assert orch.extra_metadata(args=None) == {}


def test_extra_step_logs_merges_all():
    orch = _TestOrchestrator()
    orch.add_extension(_DictExt())
    orch.add_extension(_DictExt2())
    result = orch.extra_step_logs(args=None, logs={})
    assert result == {"log_a": 1.0, "log_b": 2.0}


# --- Chain hook tests ---

def test_extra_trainable_params_chains():
    orch = _TestOrchestrator()
    orch.add_extension(_ChainExt())
    orch.add_extension(_ChainExt2())
    result = orch.extra_trainable_params(
        args=None, accelerator=None, network=None, transformer=None,
        trainable_params=["base"],
    )
    assert result == ["base", "param_a", "param_b"]


def test_extra_trainable_params_passthrough_no_extensions():
    orch = _TestOrchestrator()
    result = orch.extra_trainable_params(
        args=None, accelerator=None, network=None, transformer=None,
        trainable_params=["base"],
    )
    assert result == ["base"]


# --- Contended hook tests ---

def test_process_batch_raises_when_extension_registered():
    orch = _TestOrchestrator()
    orch.add_extension(_ProcessBatchExt())
    with pytest.raises(NotImplementedError, match="process_batch"):
        orch.process_batch()


def test_compute_loss_raises_when_extension_registered():
    orch = _TestOrchestrator()
    orch.add_extension(_ComputeLossExt())
    with pytest.raises(NotImplementedError, match="compute_loss"):
        orch.compute_loss()
