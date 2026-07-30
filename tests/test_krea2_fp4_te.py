"""Tests for --fp4_te wiring in krea2_utils.load_krea2_dit."""

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
pytest.importorskip("transformer_engine.pytorch", reason="transformer_engine not installed")

from musubi_tuner.krea2 import krea2_utils
from musubi_tuner.krea2.krea2_mmdit import SingleMMDiTConfig


def _tiny_config():
    return SingleMMDiTConfig(
        features=64, tdim=32, txtdim=32, heads=4, kvheads=2, multiplier=2,
        layers=1, patch=2, channels=4, txtheads=2, txtkvheads=2, txtlayers=1,
    )


@requires_cuda
def test_load_krea2_dit_fp4_te_swaps_block_linears(tmp_path):
    from safetensors.torch import save_file
    from musubi_tuner.krea2.krea2_mmdit import SingleStreamDiT

    config = _tiny_config()
    with torch.device("meta"):
        ref = SingleStreamDiT(config, attn_mode="torch", split_attn=False)
    sd = {k: torch.randn(v.shape, dtype=torch.bfloat16) for k, v in ref.state_dict().items()}
    path = tmp_path / "tiny.safetensors"
    save_file(sd, str(path))

    dit = krea2_utils.load_krea2_dit(
        str(path),
        device="cuda",
        dtype=torch.bfloat16,
        config=config,
        fp4_te=True,
        attn_mode="torch",
    )

    found_te_linear = False
    for name, module in dit.named_modules():
        if "blocks." in name and type(module).__module__.startswith("transformer_engine"):
            found_te_linear = True
            break
    assert found_te_linear, "expected at least one te.Linear under dit.blocks after fp4_te swap"


def test_load_krea2_dit_mutual_exclusion_fp4_te_fp8_scaled(tmp_path):
    with pytest.raises(AssertionError):
        krea2_utils.load_krea2_dit(
            str(tmp_path / "nonexistent.safetensors"),
            fp8_scaled=True,
            fp4_te=True,
        )


def test_load_krea2_dit_mutual_exclusion_fp4_te_convrot(tmp_path):
    with pytest.raises(AssertionError):
        krea2_utils.load_krea2_dit(
            str(tmp_path / "nonexistent.safetensors"),
            convrot_int8=True,
            fp4_te=True,
        )


# ---------------------------------------------------------------------------
# trainer flag validation
# ---------------------------------------------------------------------------

from types import SimpleNamespace


def _trainer_args(**overrides):
    base = dict(
        fp8_base=False,
        fp8_scaled=False,
        convrot_int8=False,
        convrot_int8_bwd="bf16",
        fp4_te=False,
        turbo_dit=None,
        turbo_dit_cache=False,
        blocks_to_swap=0,
        sample_prompts=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _handle_args(args):
    from musubi_tuner.krea2_train_network import Krea2NetworkTrainer

    trainer = Krea2NetworkTrainer()
    trainer.handle_model_specific_args(args)


def test_trainer_rejects_fp4_te_with_fp8():
    with pytest.raises(ValueError, match="fp4_te"):
        _handle_args(_trainer_args(fp4_te=True, fp8_base=True, fp8_scaled=True))


def test_trainer_rejects_fp4_te_with_convrot():
    with pytest.raises(ValueError, match="fp4_te"):
        _handle_args(_trainer_args(fp4_te=True, convrot_int8=True))


def test_trainer_rejects_fp4_te_with_turbo():
    with pytest.raises(ValueError, match="turbo"):
        _handle_args(_trainer_args(fp4_te=True, turbo_dit="turbo.safetensors"))


def test_trainer_accepts_fp4_te_alone():
    _handle_args(_trainer_args(fp4_te=True))
