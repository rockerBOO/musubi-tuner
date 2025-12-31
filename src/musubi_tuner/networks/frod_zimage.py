# frod_zimage.py
# FRoD module for Z-Image architecture

import ast
from typing import Dict, List, Optional
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Import from the main frod module
from . import frod

ZIMAGE_TARGET_REPLACE_MODULES = ["ZImageTransformerBlock"]


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> frod.FRoDNetwork:
    """
    Create FRoD network for Z-Image architecture.

    Args:
        multiplier: Output scaling factor
        network_dim: Not used in FRoD (kept for API compatibility)
        network_alpha: Alpha scaling factor
        vae: VAE module (not modified)
        text_encoders: Text encoder modules
        unet: U-Net/DiT module to apply FRoD to
        neuron_dropout: Dropout probability
        **kwargs: Additional arguments including:
            - sparse_rate: Sparsity rate for S matrix (default: 0.02)
            - regularization_alpha: Regularization for HJD (default: 1e-3)
            - exclude_patterns: List of regex patterns to exclude
            - include_patterns: List of regex patterns to include
            - verbose: Whether to print detailed info

    Returns:
        FRoDNetwork instance
    """
    # Add default exclude patterns for Z-Image
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    elif isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)

    # Exclude modulation and refiner layers (similar to LoRA version)
    exclude_patterns.append(r".*(_modulation|_refiner).*")
    kwargs["exclude_patterns"] = exclude_patterns

    return frod.create_network(
        ZIMAGE_TARGET_REPLACE_MODULES,
        "frod_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: dict[str, torch.Tensor],
    text_encoders: List[nn.Module] | None = None,
    unet: nn.Module | None = None,
    for_inference: bool = False,
    **kwargs,
) -> frod.FRoDNetwork:
    """
    Create FRoD network for Z-Image from saved weights.

    Args:
        multiplier: Output scaling factor
        weights_sd: State dict with FRoD weights
        text_encoders: Text encoder modules
        unet: U-Net/DiT module
        for_inference: Whether to use inference-optimized module
        **kwargs: Additional arguments

    Returns:
        FRoDNetwork instance with loaded weights
    """
    return frod.create_network_from_weights(
        ZIMAGE_TARGET_REPLACE_MODULES,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        **kwargs,
    )


def convert_frod_to_lora(
    frod_weights_path: str,
    output_path: str,
    rank: int = 64,
    dtype: torch.dtype = torch.float16,
) -> None:
    """
    Utility function to convert saved FRoD weights to LoRA format.

    Args:
        frod_weights_path: Path to saved FRoD weights (.safetensors or .pt)
        output_path: Output path for LoRA weights
        rank: Target LoRA rank
        dtype: Data type for output weights
    """
    import os

    # Load FRoD weights
    if frod_weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        frod_sd = load_file(frod_weights_path)
    else:
        frod_sd = torch.load(frod_weights_path, map_location="cpu")

    # Group weights by module
    modules = {}
    for key, value in frod_sd.items():
        parts = key.split(".")
        module_name = parts[0]
        param_name = ".".join(parts[1:])

        if module_name not in modules:
            modules[module_name] = {}
        modules[module_name][param_name] = value

    # Convert each module
    lora_sd = {}

    for module_name, params in modules.items():
        if "sigma" not in params or "S_values" not in params:
            continue

        sigma = params["sigma"]
        S_values = params["S_values"]
        U = params.get("U", torch.eye(S_values.shape[0]))
        V = params.get("V", torch.eye(S_values.shape[0]))
        sparse_mask = params.get("sparse_mask", torch.ones_like(S_values))
        original_weight = params.get("original_weight", torch.zeros(U.shape[0], V.shape[0]))

        # Compute delta weight
        S = S_values * sparse_mask
        Sigma = torch.diag(sigma)
        merged_weight = U @ (Sigma + S) @ V.T
        delta_w = merged_weight - original_weight

        # SVD decomposition
        U_svd, S_svd, Vh = torch.linalg.svd(delta_w.float(), full_matrices=False)

        # Truncate to rank
        r = min(rank, len(S_svd))
        S_sqrt = torch.sqrt(S_svd[:r])

        lora_down = S_sqrt.unsqueeze(1) * Vh[:r, :]
        lora_up = U_svd[:, :r] * S_sqrt.unsqueeze(0)

        lora_sd[f"{module_name}.lora_down.weight"] = lora_down.to(dtype)
        lora_sd[f"{module_name}.lora_up.weight"] = lora_up.to(dtype)
        lora_sd[f"{module_name}.alpha"] = torch.tensor(float(r))

    # Save
    if output_path.endswith(".safetensors"):
        from safetensors.torch import save_file

        save_file(lora_sd, output_path, {"converted_from": "frod", "rank": str(rank)})
    else:
        torch.save(lora_sd, output_path)

    logger.info(f"Converted FRoD to LoRA (rank={rank}): {output_path}")
