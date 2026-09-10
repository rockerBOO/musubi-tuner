import json

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.modules.mixed_quant_utils import MixedQuantizer
from musubi_tuner.modules.comfy_quant_utils import FORMAT_CONVROT_INT8, FORMAT_INT8_TENSORWISE, FORMAT_NVFP4
from musubi_tuner.modules.convrot_int8_kernels import quantize_int8_convrot_weight
from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Quantizer
from musubi_tuner.modules.nvfp4_utils import NvFp4Quantizer, _quantize_nvfp4_2d


def _nvfp4_payload():
    return torch.tensor(list(b'{"format":"nvfp4"}'), dtype=torch.uint8)


def _convrot_payload(groupsize=256):
    raw = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": groupsize}).encode("utf-8")
    return torch.tensor(list(raw), dtype=torch.uint8)


def _make_mixed_artifact(tmp_path, *, include_passthrough_bias=True):
    torch.manual_seed(0)
    nvfp4_weight = torch.randn(64, 32) * 0.02
    packed, block_scale, tensor_scale, _ = _quantize_nvfp4_2d(nvfp4_weight)

    convrot_weight = torch.randn(64, 256) * 0.02
    q_weight, q_scale = quantize_int8_convrot_weight(convrot_weight, 256)

    tensors = {
        "mlp_proj.weight": packed,
        "mlp_proj.weight_scale": block_scale,
        "mlp_proj.weight_scale_2": tensor_scale,
        "mlp_proj.comfy_quant": _nvfp4_payload(),
        "attn_proj.weight": q_weight,
        "attn_proj.weight_scale": q_scale,
        "attn_proj.comfy_quant": _convrot_payload(),
    }
    if include_passthrough_bias:
        tensors["norm.weight"] = torch.ones(64, dtype=torch.bfloat16)
    path = tmp_path / "mixed.safetensors"
    save_file(tensors, str(path))
    return str(path)


def _make_nvfp4_convrot_quantizer() -> MixedQuantizer:
    """The composition this plan wires into load_krea2_dit in Task 4 -- exercised here
    directly so this test doesn't depend on krea2_utils."""
    return MixedQuantizer(
        {
            "nvfp4": NvFp4Quantizer(foreign_formats={FORMAT_CONVROT_INT8}),
            "convrot_int8": ConvRotInt8Quantizer(target_layer_keys=[], foreign_formats={FORMAT_NVFP4, FORMAT_INT8_TENSORWISE}),
        }
    )


def test_load_and_quantize_routes_each_module_to_its_own_format(tmp_path):
    path = _make_mixed_artifact(tmp_path)
    quantizer = _make_nvfp4_convrot_quantizer()

    state_dict = quantizer.load_and_quantize([path], None)

    # NVFP4 module converted to the Musubi layout
    assert state_dict["mlp_proj.weight"].dtype is torch.uint8
    assert "mlp_proj.nvfp4_block_scale" in state_dict
    assert "mlp_proj.nvfp4_scale" in state_dict
    assert quantizer.quantizers["nvfp4"].nvfp4_module_shapes == {"mlp_proj": (64, 32)}

    # ConvRot INT8 module converted to the Musubi layout
    assert state_dict["attn_proj.weight"].dtype is torch.int8
    assert state_dict["attn_proj.scale_weight"].shape == (64, 1)
    assert quantizer.quantizers["convrot_int8"].module_groupsizes == {"attn_proj": 256}

    # passthrough tensor present exactly once, unchanged
    assert state_dict["norm.weight"].dtype is torch.bfloat16

    # no leftover ComfyUI-only keys
    assert "mlp_proj.comfy_quant" not in state_dict
    assert "attn_proj.comfy_quant" not in state_dict
    assert "attn_proj.weight_scale" not in state_dict


def test_load_and_quantize_rejects_lora_merge_hook(tmp_path):
    path = _make_mixed_artifact(tmp_path)
    quantizer = _make_nvfp4_convrot_quantizer()

    with pytest.raises(ValueError, match="Cannot merge LoRA"):
        quantizer.load_and_quantize([path], None, weight_hook=lambda key, value, keep_on_calc_device=False: value)


def test_load_and_quantize_still_rejects_a_genuinely_unsupported_format(tmp_path):
    tensors = {
        "weird.weight": torch.zeros(8, 8, dtype=torch.int8),
        "weird.weight_scale": torch.ones(8, 1, dtype=torch.float32),
        "weird.comfy_quant": torch.tensor(list(b'{"format":"awq_int4"}'), dtype=torch.uint8),
    }
    path = tmp_path / "bad.safetensors"
    save_file(tensors, str(path))
    quantizer = _make_nvfp4_convrot_quantizer()

    with pytest.raises(ValueError, match="Unsupported comfy_quant format"):
        quantizer.load_and_quantize([str(path)], None)


def test_composer_is_format_agnostic_with_a_single_sub_quantizer(tmp_path):
    """The composer itself has no NVFP4/ConvRot-specific logic -- one sub-quantizer,
    or a future third one, works the same way."""
    path = _make_mixed_artifact(tmp_path, include_passthrough_bias=False)
    quantizer = MixedQuantizer({"nvfp4": NvFp4Quantizer(foreign_formats={FORMAT_CONVROT_INT8})})

    state_dict = quantizer.load_and_quantize([path], None)

    assert "mlp_proj.nvfp4_block_scale" in state_dict
    assert "attn_proj.weight" not in state_dict  # no convrot sub-quantizer registered to claim it


def test_load_nvfp4_convrot_mixed_state_dict_returns_sub_quantizers(tmp_path):
    from musubi_tuner.modules.mixed_quant_utils import load_nvfp4_convrot_mixed_state_dict

    path = _make_mixed_artifact(tmp_path)

    state_dict, nvfp4_quantizer, convrot_quantizer = load_nvfp4_convrot_mixed_state_dict(
        [path],
        convrot_target_keys=["attn_proj"],
        convrot_exclude_keys=[],
        convrot_allowed_groupsizes=(256,),
        calc_device=None,
    )

    assert state_dict["mlp_proj.weight"].dtype is torch.uint8
    assert state_dict["attn_proj.weight"].dtype is torch.int8
    assert nvfp4_quantizer.nvfp4_module_shapes == {"mlp_proj": (64, 32)}
    assert convrot_quantizer.module_groupsizes == {"attn_proj": 256}


def test_apply_nvfp4_convrot_mixed_monkey_patch_applies_both_patches_and_freezes(tmp_path):
    import torch.nn as nn

    from musubi_tuner.modules.mixed_quant_utils import (
        apply_nvfp4_convrot_mixed_monkey_patch,
        load_nvfp4_convrot_mixed_state_dict,
    )

    path = _make_mixed_artifact(tmp_path, include_passthrough_bias=False)
    state_dict, nvfp4_quantizer, convrot_quantizer = load_nvfp4_convrot_mixed_state_dict(
        [path],
        convrot_target_keys=["attn_proj"],
        convrot_exclude_keys=[],
        convrot_allowed_groupsizes=(256,),
        calc_device=None,
    )

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp_proj = nn.Linear(32, 64, bias=False)
            self.attn_proj = nn.Linear(256, 64, bias=False)

    model = _Model()
    apply_nvfp4_convrot_mixed_monkey_patch(
        model,
        state_dict,
        nvfp4_quantizer,
        convrot_quantizer,
        convrot_bwd_mode="bf16",
        nvfp4_training=True,
        nvfp4_calc_device=torch.device("cpu"),
        nvfp4_columnwise_chunk_rows=1024,
    )

    from musubi_tuner.modules.convrot_int8_utils import block_has_convrot_patched_linear
    from musubi_tuner.modules.nvfp4_utils import block_has_nvfp4_patched_linear

    assert block_has_nvfp4_patched_linear(model.mlp_proj)
    assert block_has_convrot_patched_linear(model.attn_proj)
    assert model.is_nvfp4 is True
    assert model.is_convrot_int8 is True
    assert all(not parameter.requires_grad for parameter in model.parameters())
