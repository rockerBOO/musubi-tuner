"""Transformer Engine NVFP4 module swap for frozen base weights during LoRA training.

Unlike fp8_optimization_utils.py / convrot_int8_utils.py, this does NOT keep the
module a plain nn.Linear with a monkey-patched forward — te.Linear is a different
module class with its own parameter layout, so the child module instance is
replaced outright. This is safe for LoRA-targeting and block-swap's string-based
class checks: transformer_engine.pytorch.Linear.__name__ == "Linear" too (verified
in notes/te-fp4-build-blackwell.md / notes/nvfp4-te-implementation.md).

transformer_engine is imported lazily inside these functions so that normal
(non---fp4_te) runs never need it installed -- it is not a project dependency,
only present in the dev venv it was built into (see notes/te-fp4-build-blackwell.md).
"""

import contextlib
from typing import List

import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)


def swap_linears_to_te(module: nn.Module, target_keys: List[str], exclude_keys: List[str]) -> None:
    """Replace scoped nn.Linear children of ``module`` with te.Linear, in place.

    Scoping matches KREA2_FP8_OPTIMIZATION_TARGET_KEYS / _EXCLUDE_KEYS: a child's
    dotted name must contain any of ``target_keys`` and none of ``exclude_keys``.
    Weight/bias are copied byte-for-byte (same device/dtype) into the new module.
    """
    import transformer_engine.pytorch as tep

    swapped = 0
    for parent_name, parent in list(module.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not any(k in full_name for k in target_keys):
                continue
            if any(k in full_name for k in exclude_keys):
                continue

            te_linear = tep.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                params_dtype=torch.bfloat16,
                device=child.weight.device,
            )
            with torch.no_grad():
                te_linear.weight.copy_(child.weight)
                if child.bias is not None:
                    te_linear.bias.copy_(child.bias)
            setattr(parent, child_name, te_linear)
            swapped += 1

    logger.info(f"swapped {swapped} nn.Linear -> te.Linear (target_keys={target_keys}, exclude_keys={exclude_keys})")


def fp4_autocast(enabled: bool):
    """te.autocast(NVFP4BlockScaling) context if enabled, else a no-op context.

    disable_stochastic_rounding=True is required: NVFP4's stochastic-rounding
    gradient quantization path is hardware-gated to sm_100a/sm_103a (datacenter
    Blackwell) and excludes sm_120 (consumer Blackwell, e.g. RTX 50-series) in
    this TE version -- see notes/te-fp4-build-blackwell.md.
    """
    if not enabled:
        return contextlib.nullcontext()

    import transformer_engine.pytorch as tep
    from transformer_engine.common import recipe

    fp4_recipe = recipe.NVFP4BlockScaling(disable_stochastic_rounding=True)
    return tep.autocast(enabled=True, recipe=fp4_recipe)
