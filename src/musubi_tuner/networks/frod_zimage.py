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
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> frod.FRoDNetwork:
    """
    Create FRoD network for Z-Image from saved weights.
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
