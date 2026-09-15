"""Tests for gradient metrics collection (grad/norm, grad/mean_norm, grad/max)."""

import torch
import torch.nn as nn
import pytest

from musubi_tuner.training.trainer_base import NetworkTrainer


@pytest.fixture
def trainer():
    return NetworkTrainer()


def _params_with_grads(values: list[list[float]]) -> list[nn.Parameter]:
    """Create parameters with fixed gradient values for testing."""
    params = []
    for vals in values:
        p = nn.Parameter(torch.zeros(len(vals)))
        p.grad = torch.tensor(vals)
        params.append(p)
    return params


def test_grad_norm_single_param(trainer):
    """L2 norm of a single parameter's gradient is computed correctly."""
    # grads = [3, 4] → norm = 5
    params = _params_with_grads([[3.0, 4.0]])
    metrics = trainer.collect_grad_metrics(params)
    assert metrics["grad/norm"] == pytest.approx(5.0)


def test_grad_norm_multiple_params(trainer):
    """Total L2 norm across multiple parameters matches torch.nn.utils.clip_grad_norm_."""
    params = _params_with_grads([[1.0, 0.0], [0.0, 1.0]])
    metrics = trainer.collect_grad_metrics(params)
    # norm of [1,0,0,1] = sqrt(2)
    assert metrics["grad/norm"] == pytest.approx(2.0**0.5, rel=1e-5)


def test_grad_max_single_param(trainer):
    """Max absolute gradient value is the largest element."""
    params = _params_with_grads([[-5.0, 2.0, 1.0]])
    metrics = trainer.collect_grad_metrics(params)
    assert metrics["grad/max"] == pytest.approx(5.0)


def test_grad_max_across_params(trainer):
    """Max absolute gradient value is found across all parameters."""
    params = _params_with_grads([[1.0, 2.0], [3.0, -9.0]])
    metrics = trainer.collect_grad_metrics(params)
    assert metrics["grad/max"] == pytest.approx(9.0)


def test_empty_metrics_when_no_grads(trainer):
    """Returns empty dict when no parameters have gradients."""
    params = [nn.Parameter(torch.zeros(4))]  # grad is None
    metrics = trainer.collect_grad_metrics(params)
    assert metrics == {}


def test_grad_mean_norm_single_param(trainer):
    """Mean norm with one param equals its own norm."""
    params = _params_with_grads([[3.0, 4.0]])
    metrics = trainer.collect_grad_metrics(params)
    assert metrics["grad/mean_norm"] == pytest.approx(5.0)


def test_grad_mean_norm_multiple_params(trainer):
    """Mean norm is the average of per-parameter norms, not the total norm."""
    # param0 norm = 5, param1 norm = 5, mean = 5
    params = _params_with_grads([[3.0, 4.0], [4.0, 3.0]])
    metrics = trainer.collect_grad_metrics(params)
    assert metrics["grad/mean_norm"] == pytest.approx(5.0)


def test_skips_params_without_grad(trainer):
    """Parameters without .grad set are excluded from the computation."""
    p_with = nn.Parameter(torch.zeros(2))
    p_with.grad = torch.tensor([3.0, 4.0])
    p_without = nn.Parameter(torch.zeros(2))  # no grad
    metrics = trainer.collect_grad_metrics([p_with, p_without])
    assert metrics["grad/norm"] == pytest.approx(5.0)
    assert metrics["grad/max"] == pytest.approx(4.0)


def test_log_grad_metrics_flag_default_off():
    """--log_grad_metrics exists in the common parser and defaults to False."""
    from musubi_tuner.training.parser_common import setup_parser_common

    parser = setup_parser_common()
    args, _ = parser.parse_known_args([])
    assert args.log_grad_metrics is False
    args, _ = parser.parse_known_args(["--log_grad_metrics"])
    assert args.log_grad_metrics is True


def test_log_grad_metrics_per_module_flag_default_off():
    """--log_grad_metrics_per_module exists in the common parser and defaults to False."""
    from musubi_tuner.training.parser_common import setup_parser_common

    parser = setup_parser_common()
    args, _ = parser.parse_known_args([])
    assert args.log_grad_metrics_per_module is False
    args, _ = parser.parse_known_args(["--log_grad_metrics_per_module"])
    assert args.log_grad_metrics_per_module is True


def test_log_grad_metrics_block_regex_default_none():
    """--log_grad_metrics_block_regex defaults to None."""
    from musubi_tuner.training.parser_common import setup_parser_common

    parser = setup_parser_common()
    args, _ = parser.parse_known_args([])
    assert args.log_grad_metrics_block_regex is None


def test_log_grad_metrics_block_regex_compiles_valid_pattern():
    """A regex with exactly one capture group is compiled and stored as a re.Pattern."""
    import re

    from musubi_tuner.training.parser_common import setup_parser_common

    parser = setup_parser_common()
    args, _ = parser.parse_known_args(["--log_grad_metrics_block_regex", r"_blocks_(\d+)_"])
    assert isinstance(args.log_grad_metrics_block_regex, re.Pattern)
    assert args.log_grad_metrics_block_regex.groups == 1


def test_log_grad_metrics_block_regex_rejects_zero_groups():
    """A regex with no capture group fails at parse time, not mid-run."""
    from musubi_tuner.training.parser_common import setup_parser_common

    parser = setup_parser_common()
    with pytest.raises(SystemExit):
        parser.parse_known_args(["--log_grad_metrics_block_regex", r"_blocks_\d+_"])


def test_log_grad_metrics_block_regex_rejects_multiple_groups():
    """A regex with more than one capture group fails at parse time."""
    from musubi_tuner.training.parser_common import setup_parser_common

    parser = setup_parser_common()
    with pytest.raises(SystemExit):
        parser.parse_known_args(["--log_grad_metrics_block_regex", r"(blocks)_(\d+)"])


def _named_params_with_grads(named_values: dict[str, list[float]]) -> list[tuple[str, nn.Parameter]]:
    """Create (name, param) pairs with fixed gradient values, matching what
    network.named_parameters() yields."""
    named_params = []
    for name, vals in named_values.items():
        p = nn.Parameter(torch.zeros(len(vals)))
        p.grad = torch.tensor(vals)
        named_params.append((name, p))
    return named_params


def test_collect_grad_metrics_by_module_two_modules(trainer):
    """Each module gets its own grad/module/<name> entry, not a global total."""
    named_params = _named_params_with_grads(
        {
            "lora_unet_first.lora_down.weight": [3.0, 4.0],  # norm = 5
            "lora_unet_last_linear.lora_down.weight": [1.0, 0.0],  # norm = 1
        }
    )
    metrics = trainer.collect_grad_metrics_by_module(named_params)
    assert metrics["grad/module/lora_unet_first"] == pytest.approx(5.0)
    assert metrics["grad/module/lora_unet_last_linear"] == pytest.approx(1.0)
    assert "grad/module/lora_unet_first.lora_down.weight" not in metrics


def test_collect_grad_metrics_by_module_combines_multiple_params_per_module(trainer):
    """A module with lora_down + lora_up combines via sqrt-sum-of-squares, not linear sum."""
    named_params = _named_params_with_grads(
        {
            "lora_unet_tmlp_0.lora_down.weight": [3.0, 0.0],  # norm = 3
            "lora_unet_tmlp_0.lora_up.weight": [0.0, 4.0],  # norm = 4
        }
    )
    metrics = trainer.collect_grad_metrics_by_module(named_params)
    # combined: sqrt(3^2 + 4^2) = 5, not 3 + 4 = 7
    assert metrics["grad/module/lora_unet_tmlp_0"] == pytest.approx(5.0)


def test_collect_grad_metrics_by_module_empty_when_no_grads(trainer):
    """Returns empty dict when no parameters have gradients."""
    p = nn.Parameter(torch.zeros(4))  # grad is None
    metrics = trainer.collect_grad_metrics_by_module([("lora_unet_first.lora_down.weight", p)])
    assert metrics == {}


def test_collect_grad_metrics_by_module_block_regex_groups_matching_modules(trainer):
    """block_regex aggregates modules whose name matches into grad/block/<id>."""
    import re

    named_params = _named_params_with_grads(
        {
            "lora_unet_blocks_5_attn_wq.lora_down.weight": [3.0, 0.0],  # norm = 3
            "lora_unet_blocks_5_attn_wk.lora_down.weight": [0.0, 4.0],  # norm = 4
        }
    )
    metrics = trainer.collect_grad_metrics_by_module(named_params, block_regex=re.compile(r"_blocks_(\d+)_"))
    # sqrt(3^2 + 4^2) = 5
    assert metrics["grad/block/5"] == pytest.approx(5.0)


def test_collect_grad_metrics_by_module_block_regex_excludes_nonmatching(trainer):
    """Modules that don't match block_regex are absent from grad/block/* but present under grad/module/*."""
    import re

    named_params = _named_params_with_grads(
        {
            "lora_unet_blocks_5_attn_wq.lora_down.weight": [3.0, 4.0],
            "lora_unet_first.lora_down.weight": [1.0, 0.0],
        }
    )
    metrics = trainer.collect_grad_metrics_by_module(named_params, block_regex=re.compile(r"_blocks_(\d+)_"))
    assert "grad/block/5" in metrics
    assert not any(k.startswith("grad/block/") and k != "grad/block/5" for k in metrics)
    assert metrics["grad/module/lora_unet_first"] == pytest.approx(1.0)
    assert "grad/module/lora_unet_first" in metrics
    assert "grad/module/lora_unet_blocks_5_attn_wq" in metrics
    # lora_unet_first has no block id, so it must not contribute to any grad/block/* entry
    assert "grad/block/lora_unet_first" not in metrics
