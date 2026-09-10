"""A generic composer for per-scheme prequantized-checkpoint loaders.

Some checkpoints declare different quantization schemes on different modules within the
same file (e.g. NVFP4 on some Linears, ConvRot INT8 on others). Musubi's per-scheme
loaders (`NvFp4Quantizer`, `ConvRotInt8Quantizer`, and any future one) each convert their
own scheme's tensors into Musubi's internal layout as soon as they read them, and raise
on any module declared under a scheme they don't own -- unless constructed with
`foreign_formats` naming the scheme(s) owned by their peers, in which case they silently
skip those modules instead.

`MixedQuantizer` has no awareness of any specific scheme or source format: it just runs
each already-configured sub-loader over the same files and merges their output state
dicts. Every module is converted by exactly the sub-loader that owns it, so the merge is
lossless. Adding a new co-resident scheme means constructing its own loader with
`foreign_formats` covering its peers and adding one entry to the dict passed into this
class -- the class itself never changes.
"""

from typing import Dict, List, Optional, Union

import torch

from musubi_tuner.utils.safetensors_utils import WeightTransformHooks


class MixedQuantizer:
    """Composes named, already-configured per-format quantizers into one loader.

    ``quantizers`` maps an arbitrary caller-chosen name (e.g. ``"nvfp4"``,
    ``"convrot_int8"``) to a quantizer instance implementing the same
    ``load_and_quantize`` protocol as ``NvFp4Quantizer`` / ``ConvRotInt8Quantizer``. Each
    quantizer must already be constructed with ``foreign_formats`` covering every *other*
    entry's owned format(s), so it skips (rather than raises on) modules it doesn't own.

    After ``load_and_quantize`` returns, reach into ``self.quantizers[name]`` for that
    sub-quantizer's own format-specific results (e.g.
    ``mixed.quantizers["nvfp4"].nvfp4_module_shapes``,
    ``mixed.quantizers["convrot_int8"].module_groupsizes``) to drive the matching
    ``apply_*_monkey_patch`` call.
    """

    def __init__(self, quantizers: Dict[str, object]):
        if not quantizers:
            raise ValueError("MixedQuantizer requires at least one sub-quantizer")
        self.quantizers = quantizers

    def load_and_quantize(
        self,
        model_files: List[str],
        calc_device: Union[str, torch.device, None],
        move_to_device: bool = False,
        weight_hook: Optional[callable] = None,
        disable_numpy_memmap: bool = False,
        weight_transform_hooks: Optional[WeightTransformHooks] = None,
    ) -> Dict[str, torch.Tensor]:
        if weight_hook is not None:
            raise ValueError(
                "Cannot merge LoRA weights into a mixed-format prequantized checkpoint."
                " Use the original BF16 weights to merge LoRA at load time, or apply the LoRA at runtime."
                " / 混在フォーマットの事前量子化済みチェックポイントにはLoRAをマージできません。"
                "BF16の元重みを使用してロード時マージするか、LoRAを実行時適用してください。"
            )
        state_dict: Dict[str, torch.Tensor] = {}
        for quantizer in self.quantizers.values():
            sub_state_dict = quantizer.load_and_quantize(
                model_files,
                calc_device,
                move_to_device=move_to_device,
                disable_numpy_memmap=disable_numpy_memmap,
                weight_transform_hooks=weight_transform_hooks,
            )
            state_dict.update(sub_state_dict)
        return state_dict


def load_nvfp4_convrot_mixed_state_dict(
    model_files: List[str],
    *,
    convrot_target_keys: List[str],
    convrot_exclude_keys: List[str],
    convrot_allowed_groupsizes: tuple,
    calc_device: Union[str, torch.device, None],
    move_to_device: bool = False,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
):
    """Load and quantize a checkpoint whose Linears mix NVFP4 and ConvRot INT8 (each
    module's format declared in its own ``.comfy_quant`` spec), via ``MixedQuantizer``.

    This is the concrete NVFP4+ConvRot case of the format-agnostic ``MixedQuantizer``
    above, factored out because three call sites (Krea2, Flux.2, and now MiniMax-H3)
    independently need the exact same two-sub-quantizer wiring. Krea2 and Flux.2 still
    carry their own inline copies of this wiring as of this helper's introduction; only
    MiniMax-H3 calls this function. A later change may migrate them onto it.

    Returns ``(state_dict, nvfp4_quantizer, convrot_quantizer)``. The caller applies the
    matching monkey patches itself (see ``apply_nvfp4_convrot_mixed_monkey_patch``) since
    patch application runs against the caller's own ``nn.Module`` tree.
    """
    from musubi_tuner.modules.comfy_quant_utils import FORMAT_CONVROT_INT8, FORMAT_INT8_TENSORWISE, FORMAT_NVFP4
    from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Quantizer
    from musubi_tuner.modules.nvfp4_utils import NvFp4Quantizer
    from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8

    nvfp4_quantizer = NvFp4Quantizer(foreign_formats={FORMAT_CONVROT_INT8})
    convrot_quantizer = ConvRotInt8Quantizer(
        convrot_target_keys,
        convrot_exclude_keys,
        allowed_groupsizes=convrot_allowed_groupsizes,
        foreign_formats={FORMAT_NVFP4, FORMAT_INT8_TENSORWISE},
    )
    quantizer = MixedQuantizer({"nvfp4": nvfp4_quantizer, "convrot_int8": convrot_quantizer})
    state_dict = load_safetensors_with_lora_and_fp8(
        model_files=model_files,
        lora_weights_list=None,
        lora_multipliers=None,
        fp8_optimization=False,
        calc_device=calc_device,
        move_to_device=move_to_device,
        dit_weight_dtype=None,
        disable_numpy_memmap=disable_numpy_memmap,
        weight_transform_hooks=weight_transform_hooks,
        quantizer=quantizer,
    )
    return state_dict, nvfp4_quantizer, convrot_quantizer


def apply_nvfp4_convrot_mixed_monkey_patch(
    model: torch.nn.Module,
    sd: Dict[str, torch.Tensor],
    nvfp4_quantizer,
    convrot_quantizer,
    *,
    convrot_bwd_mode: str,
    nvfp4_training: bool,
    nvfp4_calc_device: Union[str, torch.device, None],
    nvfp4_columnwise_chunk_rows: int,
    nvfp4_use_scaled_mm: bool = True,
) -> None:
    """Apply both quantized-Linear monkey patches produced by
    ``load_nvfp4_convrot_mixed_state_dict`` in sequence.

    Safe to apply in either order: each module is owned by exactly one format (each
    Linear's format is declared once, in the checkpoint's own ``.comfy_quant`` spec), so
    this never double-patches a module. Freezes the model afterward (int8/uint8 tensors
    cannot be wrapped as ``requires_grad=True`` Parameters, and
    ``load_state_dict(assign=True)`` re-wraps incoming tensors with the meta params'
    ``requires_grad``, so the caller's subsequent strict/assign load needs the model
    already frozen).
    """
    from musubi_tuner.modules.convrot_int8_utils import apply_convrot_int8_monkey_patch
    from musubi_tuner.modules.nvfp4_utils import apply_nvfp4_monkey_patch

    apply_nvfp4_monkey_patch(
        model,
        sd,
        nvfp4_quantizer.nvfp4_module_shapes,
        nvfp4_quantizer.int8_embedding_modules,
        use_scaled_mm=nvfp4_use_scaled_mm,
        training=nvfp4_training,
        calc_device=nvfp4_calc_device,
        columnwise_chunk_rows=nvfp4_columnwise_chunk_rows,
    )
    apply_convrot_int8_monkey_patch(
        model,
        sd,
        bwd_mode=convrot_bwd_mode,
        groupsize_map=convrot_quantizer.module_groupsizes,
    )
    model.requires_grad_(False)
