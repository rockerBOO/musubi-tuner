import argparse
import time

from musubi_tuner.flux_2_profiler_train_network import Flux2ProfilerNetworkTrainer, add_profiler_args


def make_trainer_with_state(forward_ms=10, backward_ms=5, optimizer_ms=3):
    """Build a trainer instance with pre-seeded timing state."""
    trainer = Flux2ProfilerNetworkTrainer.__new__(Flux2ProfilerNetworkTrainer)
    now = time.perf_counter()
    trainer._t_step_start = now
    trainer._t_forward_end = now + forward_ms / 1000
    trainer._t_backward_start = now + forward_ms / 1000
    trainer._t_backward_end = now + (forward_ms + backward_ms) / 1000
    trainer._t_optimizer_end = now + (forward_ms + backward_ms + optimizer_ms) / 1000
    trainer.profiler = None
    return trainer


def test_extra_step_logs_returns_timing_keys():
    trainer = make_trainer_with_state()
    logs = trainer._compute_timing_logs()

    assert "profile/forward_ms" in logs
    assert "profile/backward_ms" in logs
    assert "profile/optimizer_ms" in logs
    assert "profile/step_ms" in logs


def test_timing_values_are_approximately_correct():
    trainer = make_trainer_with_state(forward_ms=10, backward_ms=5, optimizer_ms=3)
    logs = trainer._compute_timing_logs()

    assert abs(logs["profile/forward_ms"] - 10) < 1
    assert abs(logs["profile/backward_ms"] - 5) < 1
    assert abs(logs["profile/optimizer_ms"] - 3) < 1
    assert abs(logs["profile/step_ms"] - 18) < 1


def test_step_ms_equals_sum_of_phases():
    trainer = make_trainer_with_state(forward_ms=20, backward_ms=8, optimizer_ms=4)
    logs = trainer._compute_timing_logs()

    total = logs["profile/forward_ms"] + logs["profile/backward_ms"] + logs["profile/optimizer_ms"]
    assert abs(logs["profile/step_ms"] - total) < 0.01


def test_profiler_args_defaults():
    """Parser defaults match documented values."""
    parser = argparse.ArgumentParser()
    add_profiler_args(parser)
    args = parser.parse_args([])

    assert args.profile_warmup == 2
    assert args.profile_steps == 5
    assert args.profile_output_dir == "profiling"


def test_extra_step_logs_with_no_profiler_returns_timing():
    """extra_step_logs works and returns timing keys even when profiler is None."""
    trainer = make_trainer_with_state()
    assert trainer.profiler is None
    logs = trainer.extra_step_logs(argparse.Namespace(), {})

    assert "profile/forward_ms" in logs
    assert "profile/step_ms" in logs
