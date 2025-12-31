# frod.py
# FRoD network module - Full-Rank fine-tuning with Rotational Degrees of freedom
# Can be converted to LoRA format for inference compatibility

import ast
import math
import os
import re
from typing import Dict, List, Optional, Type, Union
from collections import defaultdict
import numpy as np
from numpy.linalg import qr, inv

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

HUNYUAN_TARGET_REPLACE_MODULES = ["MMDoubleStreamBlock", "MMSingleStreamBlock"]


class FRoDModule(torch.nn.Module):
    """
    FRoD module that replaces forward method of the original Linear.

    Implements: W' = U(Σ + S)Vᵀ where:
    - U: frozen approximately orthogonal matrix (per-layer)
    - Σ: trainable diagonal scaling (on-axis)
    - S: trainable sparse matrix (off-axis rotation)
    - V: frozen shared basis (across layers of same category)
    """

    def __init__(
        self,
        frod_name: str,
        org_module: torch.nn.Module,
        multiplier: float = 1.0,
        sparse_rate: float = 0.02,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        V: Optional[torch.Tensor] = None,
        sparse_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            frod_name: Name identifier for this module
            org_module: Original module to adapt
            multiplier: Output scaling factor
            sparse_rate: Sparsity rate for S matrix (fraction of non-zero off-diagonal elements)
            alpha: Scaling factor (similar to LoRA alpha)
            dropout: Standard dropout probability
            rank_dropout: Dropout on intermediate representation
            module_dropout: Probability to skip entire module
            V: Shared basis matrix (if None, will be set later via set_shared_basis)
            sparse_mask: Sparsity pattern for S (if None, will be created)
        """
        super().__init__()
        self.frod_name = frod_name

        if org_module.__class__.__name__ == "Conv2d":
            raise NotImplementedError("FRoD does not support Conv2d yet")

        in_dim = org_module.in_features
        out_dim = org_module.out_features

        self.in_features = in_dim
        self.out_features = out_dim
        self.sparse_rate = sparse_rate
        self.multiplier = multiplier

        # Store original weight
        self.register_buffer("original_weight", org_module.weight.data.clone())

        # Alpha scaling (similar to LoRA)
        if alpha is None or alpha == 0:
            alpha = in_dim
        self.scale = alpha / in_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # These will be set by set_shared_basis() after hierarchical decomposition
        self.register_buffer("U", torch.eye(out_dim, in_dim))

        self._V_key: Optional[str] = None
        self.network: Optional["FRoDNetwork"] = None

        # Create or use provided sparse mask
        if sparse_mask is not None:
            self.register_buffer("sparse_mask", sparse_mask)
        else:
            self.register_buffer("sparse_mask", self._create_sparse_mask(in_dim, sparse_rate))

        # Trainable parameters - initialized after V is set
        self.sigma = nn.Parameter(torch.ones(in_dim))  # Diagonal scaling
        self.S_values = nn.Parameter(torch.zeros(in_dim, in_dim))  # Sparse rotation

        self._initialized = False

        # Dropout settings
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        self.org_module = org_module  # Will be removed in apply_to()

    @property
    def V(self) -> torch.Tensor:
        """Get shared V from network."""
        if self._network_ref is None or self._V_key is None:
            raise RuntimeError(f"FRoD module {self.frod_name} not initialized - call set_network() first")
        return self._network_ref.shared_V[self._V_key]

    def _create_sparse_mask(self, n: int, sparsity: float) -> torch.Tensor:
        """Create random off-diagonal sparse mask."""
        rows, cols = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        mask_indices = torch.stack([rows.flatten(), cols.flatten()], dim=1)
        off_diag = mask_indices[mask_indices[:, 0] != mask_indices[:, 1]]

        k = min(int(n * n * sparsity), off_diag.shape[0])
        if k > 0:
            perm = torch.randperm(off_diag.shape[0])[:k]
            selected = off_diag[perm]
            mask = torch.zeros(n, n)
            mask[selected[:, 0], selected[:, 1]] = 1.0
        else:
            mask = torch.zeros(n, n)

        return mask

    def initialize_from_weight(self):
        """Initialize U and sigma from original weight and shared V."""
        if self._initialized:
            return

        W = self.original_weight.data
        V = self.V

        # Compute U and sigma such that W ≈ U @ diag(sigma) @ V.T
        w = W.detach().cpu().numpy()
        v = V.detach().cpu().numpy()

        try:
            v_inv_T = inv(v).T
            B = w @ v_inv_T

            sigma = np.linalg.norm(B, axis=0)
            sigma = np.where(sigma > 1e-8, sigma, 1e-8)  # Avoid division by zero
            U = B / sigma

            self.U.data = torch.from_numpy(U).to(W.dtype).to(W.device)
            self.sigma.data = torch.from_numpy(sigma).to(W.dtype).to(W.device)
        except Exception as e:
            logger.warning(f"Failed to initialize {self.frod_name}: {e}, using identity")
            # Fallback to identity-like initialization
            self.sigma.data = torch.ones(self.in_features, device=W.device, dtype=W.dtype)

        # Initialize S_values to zero (no rotation initially)
        self.S_values.data.zero_()

        self._initialized = True

    def set_shared_basis(self, V: torch.Tensor, sparse_mask: Optional[torch.Tensor] = None):
        """Set the shared basis V and optionally update sparse mask."""
        device = self.original_weight.device
        dtype = self.original_weight.dtype

        self.V = V.to(device=device, dtype=dtype)

        if sparse_mask is not None:
            self.sparse_mask = sparse_mask.to(device=device, dtype=dtype)

        # Re-initialize with new V
        self._initialized = False
        self.initialize_from_weight()

    def get_delta_weight(self) -> torch.Tensor:
        """Compute the weight delta: W_frod - W_original."""
        return self.get_merged_weight() - self.original_weight

    def get_merged_weight(self) -> torch.Tensor:
        """Compute effective weight: U(Σ + S)Vᵀ"""
        S = self.S_values * self.sparse_mask
        Sigma = torch.diag(self.sigma)
        weight = self.U @ (Sigma + S) @ self.V.T
        return weight

    def set_network(self, network: "FRoDNetwork"):
        """Set reference to parent network for accessing shared V."""
        self.network = network

    def set_category(self, category: str):
        """Set category key for shared V lookup."""
        self._V_key = category

    def apply_to(self):
        """Replace the original module's forward with this module's forward."""
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        org_forwarded = self.org_forward(x)

        # Module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        # Compute FRoD contribution
        S = self.S_values * self.sparse_mask.to(x.device)
        Sigma = torch.diag(self.sigma)

        # Delta weight
        delta_w = self.U @ (Sigma + S) @ self.V.T - self.original_weight.to(x.device)

        # Apply dropout
        if self.dropout is not None and self.training:
            x = F.dropout(x, p=self.dropout)

        # FRoD output
        frod_out = F.linear(x, delta_w)

        return org_forwarded + frod_out * self.multiplier * self.scale

    def to_lora(self, rank: int = 64, clamp_quantile: float = 0.99) -> Dict[str, torch.Tensor]:
        """
        Convert FRoD delta to LoRA format using truncated SVD.

        Args:
            rank: Target LoRA rank
            clamp_quantile: Clamp extreme values before SVD

        Returns:
            Dict with 'lora_down.weight', 'lora_up.weight', 'alpha'
        """
        delta_w = self.get_delta_weight().detach().float()

        # Clamp extreme values
        if clamp_quantile < 1.0:
            max_val = torch.quantile(delta_w.abs().flatten(), clamp_quantile)
            delta_w = delta_w.clamp(-max_val, max_val)

        # SVD decomposition
        U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)

        # Truncate to rank
        rank = min(rank, len(S))
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]

        # Split singular values (sqrt split)
        S_sqrt = torch.sqrt(S_r)

        # LoRA convention: output = x @ A^T @ B^T, so ΔW = B @ A
        # lora_down (A): (rank, in_features)
        # lora_up (B): (out_features, rank)
        lora_down = S_sqrt.unsqueeze(1) * Vh_r  # (rank, in_features)
        lora_up = U_r * S_sqrt.unsqueeze(0)  # (out_features, rank)

        # Compute reconstruction error
        reconstructed = lora_up @ lora_down
        error = (delta_w - reconstructed).norm() / (delta_w.norm() + 1e-8)

        return {
            "lora_down.weight": lora_down,
            "lora_up.weight": lora_up,
            "alpha": torch.tensor(float(rank)),  # Use rank as alpha for scale=1
            "reconstruction_error": error.item(),
        }


class FRoDInfModule(FRoDModule):
    """FRoD module optimized for inference."""

    def __init__(
        self,
        frod_name: str,
        org_module: torch.nn.Module,
        multiplier: float = 1.0,
        sparse_rate: float = 0.02,
        alpha: float = 1.0,
        **kwargs,
    ):
        # No dropout for inference
        super().__init__(frod_name, org_module, multiplier, sparse_rate, alpha)

        self.org_module_ref = [org_module]
        self.enabled = True
        self.network: "FRoDNetwork" = None
        self._cached_weight: Optional[torch.Tensor] = None


    def cache_weights(self):
        """Pre-compute merged weights for fast inference."""
        self._cached_weight = self.get_merged_weight().detach()

    def clear_cache(self):
        """Clear cached weights."""
        self._cached_weight = None

    def merge_to(self, sd: Dict[str, torch.Tensor], dtype, device, non_blocking: bool = False):
        """Merge FRoD weights into the original module."""
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"].to(device, dtype=torch.float, non_blocking=non_blocking)

        if dtype is None:
            dtype = org_sd["weight"].dtype
        if device is None:
            device = org_sd["weight"].device

        # Load FRoD parameters from state dict
        sigma = sd.get("sigma", self.sigma).to(device, torch.float, non_blocking=non_blocking)
        S_values = sd.get("S_values", self.S_values).to(device, torch.float, non_blocking=non_blocking)
        U = sd.get("U", self.U).to(device, torch.float, non_blocking=non_blocking)
        V = sd.get("V", self.V).to(device, torch.float, non_blocking=non_blocking)
        sparse_mask = sd.get("sparse_mask", self.sparse_mask).to(device, torch.float, non_blocking=non_blocking)

        # Compute merged weight
        S = S_values * sparse_mask
        Sigma = torch.diag(sigma)
        frod_weight = U @ (Sigma + S) @ V.T

        # Compute delta and apply
        original_weight = sd.get("original_weight", self.original_weight).to(device, torch.float, non_blocking=non_blocking)
        delta_w = frod_weight - original_weight
        weight = weight + self.multiplier * delta_w * self.scale

        org_sd["weight"] = weight.to(device, dtype=dtype)
        self.org_module.load_state_dict(org_sd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.org_forward(x)

        if self._cached_weight is not None:
            # Use cached weight
            delta_w = self._cached_weight - self.original_weight.to(x.device)
            return self.org_forward(x) + F.linear(x, delta_w) * self.multiplier * self.scale

        return super().forward(x)


class FRoDNetwork(torch.nn.Module):
    """
    FRoD Network that manages FRoD modules across a model.

    Supports hierarchical joint decomposition for shared basis V,
    and can convert to LoRA format for inference compatibility.
    """

    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders: Union[List[nn.Module], nn.Module],
        unet: nn.Module,
        multiplier: float = 1.0,
        sparse_rate: float = 0.02,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        module_class: Type[object] = FRoDModule,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        verbose: bool = False,
        regularization_alpha: float = 1e-3,
    ) -> None:
        super().__init__()

        self.multiplier = multiplier
        self.sparse_rate = sparse_rate
        self.alpha = alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.target_replace_modules = target_replace_modules
        self.prefix = prefix
        self.regularization_alpha = regularization_alpha
        self.verbose = verbose

        # Compile regex patterns
        self.exclude_re_patterns = []
        if exclude_patterns:
            for pattern in exclude_patterns:
                try:
                    self.exclude_re_patterns.append(re.compile(pattern))
                except re.error as e:
                    logger.error(f"Invalid exclude pattern '{pattern}': {e}")

        self.include_re_patterns = []
        if include_patterns:
            for pattern in include_patterns:
                try:
                    self.include_re_patterns.append(re.compile(pattern))
                except re.error as e:
                    logger.error(f"Invalid include pattern '{pattern}': {e}")

        # Storage for modules organized by category (for shared basis)
        self.modules_by_category: Dict[str, List[FRoDModule]] = defaultdict(list)
        self.shared_V: Dict[str, torch.Tensor] = {}
        self.shared_masks: Dict[str, torch.Tensor] = {}

        logger.info(f"Creating FRoD network. sparse_rate: {sparse_rate}, alpha: {alpha}")
        logger.info(f"dropout: {dropout}, rank_dropout: {rank_dropout}, module_dropout: {module_dropout}")

        # Create modules
        self.text_encoder_frods: List[FRoDModule] = []
        self.unet_frods: List[FRoDModule] = []

        self.unet_frods, skipped = self._create_modules(prefix, unet, target_replace_modules, module_class)

        logger.info(f"Created FRoD for U-Net/DiT: {len(self.unet_frods)} modules")

        if verbose and skipped:
            logger.warning(f"Skipped {len(skipped)} modules")
            for name in skipped:
                logger.info(f"\t{name}")

        # Verify unique names
        names = set()
        for frod in self.text_encoder_frods + self.unet_frods:
            assert frod.frod_name not in names, f"Duplicate name: {frod.frod_name}"
            names.add(frod.frod_name)

    def _get_category(self, name: str) -> str:
        """Extract category from module name for shared basis grouping."""
        # Group by the last component (e.g., 'q_proj', 'v_proj', etc.)
        parts = name.split(".")
        # Try to find meaningful category
        for part in reversed(parts):
            if part in [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "q",
                "k",
                "v",
                "o",
                "dense",
                "fc1",
                "fc2",
                "linear",
                "proj",
                "to_q",
                "to_k",
                "to_v",
                "to_out",
                "img_attn_q",
                "img_attn_k",
                "img_attn_v",
                "img_attn_proj",
                "txt_attn_q",
                "txt_attn_k",
                "txt_attn_v",
                "txt_attn_proj",
                "img_mlp_0",
                "img_mlp_2",
                "txt_mlp_0",
                "txt_mlp_2",
                "linear1",
                "linear2",
                "to_out_linear",
            ]:
                return part
        # Fallback to last part
        return parts[-1] if parts else "default"

    def _create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_mods: Optional[List[str]],
        module_class: Type[FRoDModule],
    ) -> tuple:
        """Create FRoD modules for target layers."""
        frods = []
        skipped = []

        # First, collect all candidate modules
        candidates = []
        for name, module in root_module.named_modules():
            if target_replace_mods is None or module.__class__.__name__ in target_replace_mods:
                if target_replace_mods is None:
                    module = root_module

                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ != "Linear":
                        continue

                    original_name = (name + "." if name else "") + child_name
                    frod_name = f"{prefix}.{original_name}".replace(".", "_")

                    # Check exclude/include patterns
                    excluded = any(p.match(original_name) for p in self.exclude_re_patterns)
                    included = any(p.match(original_name) for p in self.include_re_patterns)

                    if excluded and not included:
                        skipped.append(original_name)
                        continue

                    candidates.append((original_name, frod_name, child_module))

                if target_replace_mods is None:
                    break

        # Create modules with progress bar
        logger.info(f"Creating {len(candidates)} FRoD modules...")
        for original_name, frod_name, child_module in tqdm(candidates, desc="Creating FRoD modules", disable=not self.verbose):
            # Get category for shared basis
            category = self._get_category(original_name)

            frod = module_class(
                frod_name,
                child_module,
                multiplier=self.multiplier,
                sparse_rate=self.sparse_rate,
                alpha=self.alpha,
                dropout=self.dropout,
                rank_dropout=self.rank_dropout,
                module_dropout=self.module_dropout,
            )
            frod.set_network(self)

            frods.append(frod)
            self.modules_by_category[category].append(frod)

            if self.verbose:
                logger.debug(f"\t{frod_name} (category: {category})")

        return frods, skipped

    def compute_shared_bases(self):
        """
        Compute shared basis V for each category using Hierarchical Joint Decomposition.
        Should be called after all modules are created but before training.
        """
        logger.info("Computing shared bases via Hierarchical Joint Decomposition...")
        categories = list(self.modules_by_category.keys())
        
        for category in tqdm(categories, desc="Computing shared bases"):
            modules = self.modules_by_category[category]
            if not modules:
                continue
            
            # Collect weights
            weights = [m.original_weight for m in modules]
            
            # Check dimensions match
            in_dims = [w.shape[1] for w in weights]
            if len(set(in_dims)) > 1:
                logger.warning(f"  Dimension mismatch in category '{category}', skipping shared basis")
                # Initialize each module independently (fallback)
                for m in modules:
                    m._network_ref = self
                    m._V_key = None  # Will use fallback identity matrix
                    m.initialize_from_weight()
                continue
            
            # Compute shared basis
            V = self._compute_shared_basis(weights, self.regularization_alpha)
            
            # Register as buffer so it's saved/loaded with state_dict
            buffer_name = f'shared_V_{category.replace(".", "_")}'  # sanitize name for module registration
            self.register_buffer(buffer_name, V)
            self.shared_V[category] = getattr(self, buffer_name)
            
            # Create shared sparse mask
            n = V.shape[0]
            mask = self._create_sparse_mask(n, self.sparse_rate)
            mask_buffer_name = f'shared_mask_{category.replace(".", "_")}'
            self.register_buffer(mask_buffer_name, mask)
            self.shared_masks[category] = getattr(self, mask_buffer_name)
            
            # Set network reference and category key for all modules
            for m in tqdm(
                modules,
                desc=f"  Initializing '{category}'",
                leave=False,
                disable=len(modules) < 10,
            ):
                m._network_ref = self
                m._V_key = category
                m.initialize_from_weight()  # Now computes U and init_sigma using shared V
        
        logger.info(f"Shared bases computed for {len(categories)} categories")
        def _compute_shared_basis(self, weights: List[torch.Tensor], pi: float) -> torch.Tensor:
            """Compute shared basis V using Hierarchical Joint Decomposition."""
            A_list = [w.detach().cpu().numpy() for w in weights]
            A_stack = np.vstack(A_list)

            Q, R_global = qr(A_stack)

            Q_list = []
            row_idx = 0
            for A in A_list:
                m = A.shape[0]
                Q_list.append(Q[row_idx : row_idx + m, :])
                row_idx += m

            n = R_global.shape[1]
            T_pi = np.zeros((n, n), dtype=R_global.dtype)

            for Qi in Q_list:
                Qi_term = Qi.T @ Qi + pi * np.eye(n)
                T_pi += np.linalg.inv(Qi_term)
            T_pi /= len(Q_list)

            _, Z = np.linalg.eigh(T_pi)
            V = R_global.T @ Z

            return torch.from_numpy(V).float()

        def _create_sparse_mask(self, n: int, sparsity: float) -> torch.Tensor:
            """Create random off-diagonal sparse mask."""
            rows, cols = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
            mask_indices = torch.stack([rows.flatten(), cols.flatten()], dim=1)
            off_diag = mask_indices[mask_indices[:, 0] != mask_indices[:, 1]]

            k = min(int(n * n * sparsity), off_diag.shape[0])
            if k > 0:
                perm = torch.randperm(off_diag.shape[0])[:k]
                selected = off_diag[perm]
                mask = torch.zeros(n, n)
                mask[selected[:, 0], selected[:, 1]] = 1.0
            else:
                mask = torch.zeros(n, n)

            return mask

        def prepare_network(self, args=None):
            """Called after network creation, compute shared bases here."""
            self.compute_shared_bases()

        def set_multiplier(self, multiplier: float):
            self.multiplier = multiplier
            for frod in self.text_encoder_frods + self.unet_frods:
                frod.multiplier = multiplier

        def set_enabled(self, is_enabled: bool):
            for frod in self.text_encoder_frods + self.unet_frods:
                if hasattr(frod, "enabled"):
                    frod.enabled = is_enabled

        def load_weights(self, file: str):
            if os.path.splitext(file)[1] == ".safetensors":
                from safetensors.torch import load_file

                weights_sd = load_file(file)
            else:
                weights_sd = torch.load(file, map_location="cpu")

            info = self.load_state_dict(weights_sd, strict=False)
            return info

        def apply_to(
            self,
            text_encoders: Optional[nn.Module],
            unet: Optional[nn.Module],
            apply_text_encoder: bool = True,
            apply_unet: bool = True,
        ):
            if not apply_text_encoder:
                self.text_encoder_frods = []
            if not apply_unet:
                self.unet_frods = []

            if not self.text_encoder_frods and not self.unet_frods:
                raise RuntimeError("No FRoD modules found")

            logger.info(f"Applying FRoD to model...")

            all_frods = self.text_encoder_frods + self.unet_frods
            for frod in tqdm(all_frods, desc="Applying FRoD modules"):
                frod.apply_to()
                self.add_module(frod.frod_name, frod)

            logger.info(f"Applied {len(all_frods)} FRoD modules")

        def is_mergeable(self) -> bool:
            return True

        def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None, non_blocking=False):
            """Merge FRoD weights into the base model."""
            from concurrent.futures import ThreadPoolExecutor

            all_frods = self.text_encoder_frods + self.unet_frods

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                for frod in tqdm(all_frods, desc="Merging FRoD weights"):
                    sd_for_frod = {}
                    for key in weights_sd.keys():
                        if key.startswith(frod.frod_name):
                            sd_for_frod[key[len(frod.frod_name) + 1 :]] = weights_sd[key]

                    if len(sd_for_frod) == 0:
                        logger.info(f"No weight for {frod.frod_name}")
                        continue

                    futures.append(executor.submit(frod.merge_to, sd_for_frod, dtype, device, non_blocking))

                for future in futures:
                    future.result()

            logger.info("FRoD weights merged")

        def prepare_optimizer_params(self, unet_lr: float = 1e-4, **kwargs):
            """
            Prepare optimizer parameters with separate learning rates for sigma and S.

            FRoD paper recommends higher LR for sigma (on-axis) than S (off-axis).
            """
            self.requires_grad_(True)

            # Get learning rate ratio (default: sigma gets 10x the LR of S)
            sigma_lr_ratio = kwargs.get("sigma_lr_ratio", 10.0)
            s_lr = unet_lr
            sigma_lr = unet_lr * sigma_lr_ratio

            all_params = []
            lr_descriptions = []

            sigma_params = []
            s_params = []

            for frod in self.unet_frods:
                sigma_params.append(frod.sigma)
                s_params.append(frod.S_values)

            if sigma_params:
                all_params.append({"params": sigma_params, "lr": sigma_lr})
                lr_descriptions.append(f"sigma (lr={sigma_lr})")

            if s_params:
                all_params.append({"params": s_params, "lr": s_lr})
                lr_descriptions.append(f"S_values (lr={s_lr})")

            return all_params, lr_descriptions

        def get_trainable_params(self):
            return self.parameters()

        def save_weights(self, file: str, dtype, metadata: Optional[Dict] = None):
            if metadata is not None and len(metadata) == 0:
                metadata = None

            state_dict = self.state_dict()

            if dtype is not None:
                logger.info(f"Converting weights to {dtype}...")
                for key in tqdm(list(state_dict.keys()), desc="Converting dtype"):
                    v = state_dict[key]
                    v = v.detach().clone().to("cpu").to(dtype)
                    state_dict[key] = v

            if os.path.splitext(file)[1] == ".safetensors":
                from safetensors.torch import save_file

                if metadata is None:
                    metadata = {}
                metadata["frod_format"] = "true"
                metadata["sparse_rate"] = str(self.sparse_rate)
                metadata["alpha"] = str(self.alpha)
                save_file(state_dict, file, metadata)
            else:
                torch.save(state_dict, file)

            logger.info(f"Saved FRoD weights to: {file}")

        def convert_to_lora(
            self,
            rank: int = 64,
            clamp_quantile: float = 0.99,
        ) -> Dict[str, torch.Tensor]:
            """
            Convert all FRoD modules to LoRA format.

            Args:
                rank: Target LoRA rank
                clamp_quantile: Clamp extreme values before SVD

            Returns:
                State dict in LoRA format
            """
            lora_state_dict = {}

            logger.info(f"Converting FRoD to LoRA (rank={rank})...")

            all_frods = self.text_encoder_frods + self.unet_frods
            total_error = 0

            for frod in tqdm(all_frods, desc="Converting to LoRA"):
                lora_data = frod.to_lora(rank=rank, clamp_quantile=clamp_quantile)

                # Use same naming convention as LoRA
                lora_name = frod.frod_name
                lora_state_dict[f"{lora_name}.lora_down.weight"] = lora_data["lora_down.weight"]
                lora_state_dict[f"{lora_name}.lora_up.weight"] = lora_data["lora_up.weight"]
                lora_state_dict[f"{lora_name}.alpha"] = lora_data["alpha"]

                total_error += lora_data["reconstruction_error"]

            avg_error = total_error / max(len(all_frods), 1)
            logger.info(f"Conversion complete. Average reconstruction error: {avg_error:.4f}")

            return lora_state_dict

        def save_as_lora(
            self,
            file: str,
            rank: int = 64,
            dtype=None,
            metadata: Optional[Dict] = None,
            clamp_quantile: float = 0.99,
        ):
            """
            Save FRoD weights in LoRA-compatible format.

            Args:
                file: Output file path
                rank: Target LoRA rank
                dtype: Data type for saved weights
                metadata: Optional metadata dict
                clamp_quantile: Clamp extreme values before SVD
            """
            lora_state_dict = self.convert_to_lora(rank=rank, clamp_quantile=clamp_quantile)

            if dtype is not None:
                logger.info(f"Converting to {dtype}...")
                for key in tqdm(list(lora_state_dict.keys()), desc="Converting dtype"):
                    lora_state_dict[key] = lora_state_dict[key].to(dtype)

            if os.path.splitext(file)[1] == ".safetensors":
                from safetensors.torch import save_file

                if metadata is None:
                    metadata = {}
                metadata["lora_converted_from_frod"] = "true"
                metadata["frod_to_lora_rank"] = str(rank)
                save_file(lora_state_dict, file, metadata)
            else:
                torch.save(lora_state_dict, file)

            logger.info(f"Saved LoRA weights to: {file}")

        def enable_gradient_checkpointing(self):
            pass

        def prepare_grad_etc(self, unet):
            self.requires_grad_(True)

        def on_epoch_start(self, unet):
            self.train()

        def on_step_start(self):
            pass

        def backup_weights(self):
            """Backup original weights for potential restoration."""
            all_frods: List[FRoDInfModule] = self.text_encoder_frods + self.unet_frods
            for frod in tqdm(all_frods, desc="Backing up weights"):
                if hasattr(frod, "org_module_ref"):
                    org_module = frod.org_module_ref[0]
                    if not hasattr(org_module, "_frod_org_weight"):
                        sd = org_module.state_dict()
                        org_module._frod_org_weight = sd["weight"].detach().clone()
                        org_module._frod_restored = True

        def restore_weights(self):
            """Restore original weights."""
            all_frods: List[FRoDInfModule] = self.text_encoder_frods + self.unet_frods
            for frod in tqdm(all_frods, desc="Restoring weights"):
                if hasattr(frod, "org_module_ref"):
                    org_module = frod.org_module_ref[0]
                    if hasattr(org_module, "_frod_org_weight") and not org_module._frod_restored:
                        sd = org_module.state_dict()
                        sd["weight"] = org_module._frod_org_weight
                        org_module.load_state_dict(sd)
                        org_module._frod_restored = True

        def pre_calculation(self):
            """Pre-calculate merged weights for inference."""
            all_frods: List[FRoDInfModule] = self.text_encoder_frods + self.unet_frods
            for frod in tqdm(all_frods, desc="Pre-calculating weights"):
                if hasattr(frod, "org_module_ref"):
                    org_module = frod.org_module_ref[0]
                    sd = org_module.state_dict()

                    org_weight = sd["weight"]
                    delta_weight = frod.get_delta_weight().to(org_weight.device, dtype=org_weight.dtype)
                    sd["weight"] = org_weight + delta_weight * frod.multiplier * frod.scale
                    org_module.load_state_dict(sd)

                    org_module._frod_restored = False
                    frod.enabled = False


def create_network(
    target_replace_modules: List[str],
    prefix: str,
    multiplier: float,
    network_dim: Optional[int],  # Not used in FRoD, kept for API compatibility
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> FRoDNetwork:
    """Create FRoD network - API compatible with LoRA create_network."""

    sparse_rate = kwargs.get("sparse_rate", 0.02)
    regularization_alpha = kwargs.get("regularization_alpha", 1e-3)

    if network_alpha is None:
        network_alpha = 1.0

    # Parse kwargs
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)

    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    verbose = kwargs.get("verbose", False)
    if isinstance(verbose, str):
        verbose = verbose.lower() == "true"

    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is not None and isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)

    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None and isinstance(include_patterns, str):
        include_patterns = ast.literal_eval(include_patterns)

    network = FRoDNetwork(
        target_replace_modules,
        prefix,
        text_encoders,
        unet,
        multiplier=multiplier,
        sparse_rate=sparse_rate,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        verbose=verbose,
        regularization_alpha=regularization_alpha,
    )

    return network


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> FRoDNetwork:
    """Create FRoD network for HunyuanVideo architecture."""

    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    elif isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)

    exclude_patterns.append(r".*(img_mod|txt_mod|modulation).*")
    kwargs["exclude_patterns"] = exclude_patterns

    return create_network(
        HUNYUAN_TARGET_REPLACE_MODULES,
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


def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> FRoDNetwork:
    """Create FRoD network from saved weights."""

    module_class = FRoDInfModule if for_inference else FRoDModule

    network = FRoDNetwork(
        target_replace_modules,
        "frod_unet",
        text_encoders,
        unet,
        multiplier=multiplier,
        module_class=module_class,
        **kwargs,
    )

    return network


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> FRoDNetwork:
    """Create FRoD network for HunyuanVideo from saved weights."""

    return create_network_from_weights(
        HUNYUAN_TARGET_REPLACE_MODULES,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        **kwargs,
    )
