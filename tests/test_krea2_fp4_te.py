"""Tests for --fp4_te wiring in krea2_utils.load_krea2_dit."""

from types import SimpleNamespace

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _te_available() -> bool:
    try:
        import transformer_engine.pytorch  # noqa: F401

        return True
    except ImportError:
        return False


# Only the one test that actually performs the fp4_te swap needs this -- the mutual-exclusion
# and trainer-flag-validation tests below must run everywhere transformer_engine isn't
# installed too, since none of them touch the swap path.
requires_te = pytest.mark.skipif(not _te_available(), reason="transformer_engine not installed")

from musubi_tuner.krea2 import krea2_utils
from musubi_tuner.krea2.krea2_mmdit import SingleMMDiTConfig


def _tiny_config():
    return SingleMMDiTConfig(
        features=64, tdim=32, txtdim=32, heads=4, kvheads=2, multiplier=2,
        layers=1, patch=2, channels=4, txtheads=2, txtkvheads=2, txtlayers=1,
    )


@requires_cuda
@requires_te
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
    # Gates the gradient-checkpointing context_fn fix (krea2_mmdit.py) -- if this regresses to
    # False, checkpointed training under fp4_te silently stops reactivating FP4 autocast during
    # recompute and CheckpointError comes back.
    assert dit.fp4_te is True


def _tiny_forward(model, B=1, H=8, W=8, n_txt=3):
    """Drives model.forward directly, matching tests/test_self_flow_k2_features.py's pattern."""
    from einops import repeat

    patch = model.config.patch
    h_, w_ = H // patch, W // patch
    imglen = h_ * w_
    img = torch.randn(B, imglen, model.config.channels * patch**2)
    context = torch.randn(B, n_txt, model.config.txtlayers, model.config.txtdim)
    imgids = torch.zeros((h_, w_, 3))
    imgids[..., 1] = torch.arange(h_)[:, None]
    imgids[..., 2] = torch.arange(w_)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=B, three=3)
    txtpos = torch.zeros(B, n_txt, 3)
    pos = torch.cat((imgpos, txtpos), dim=1)
    mask = torch.ones(B, imglen + n_txt, dtype=torch.bool)
    model(img=img, context=context, t=torch.rand(B), pos=pos, mask=mask)


def test_checkpoint_call_only_passes_context_fn_when_fp4_te(tiny_k2_config, monkeypatch):
    """CPU-only wiring check for the production checkpoint fix -- no CUDA/TE needed, since
    it never actually runs checkpointed recompute (checkpoint() is monkeypatched to call the
    block directly). Verifies: context_fn is passed to torch.utils.checkpoint.checkpoint only
    when dit.fp4_te is True, and never when it's False (matching the fix for the review finding
    that passing a non-default context_fn unconditionally breaks
    torch.utils.checkpoint.set_checkpoint_debug_enabled(True) and dynamo tracing for every K2
    run, not just fp4_te ones)."""
    import musubi_tuner.krea2.krea2_mmdit as krea2_mmdit_module

    model = krea2_mmdit_module.SingleStreamDiT(tiny_k2_config, attn_mode="torch")
    model.enable_gradient_checkpointing()
    model.train()

    captured_kwargs = []

    def fake_checkpoint(fn, *args, **kwargs):
        captured_kwargs.append(kwargs)
        return fn(*args)

    monkeypatch.setattr(krea2_mmdit_module.torch.utils.checkpoint, "checkpoint", fake_checkpoint)

    model.fp4_te = False
    _tiny_forward(model)
    assert len(captured_kwargs) == len(model.blocks)
    assert all("context_fn" not in kw for kw in captured_kwargs), "fp4_te=False must not pass context_fn"

    captured_kwargs.clear()
    model.fp4_te = True
    _tiny_forward(model)
    assert len(captured_kwargs) == len(model.blocks)
    assert all("context_fn" in kw for kw in captured_kwargs), "fp4_te=True must pass context_fn"


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
