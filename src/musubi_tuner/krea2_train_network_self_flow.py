"""Self-Flow training entry point for Krea 2 (K2).

Implements Self-Supervised Flow Matching (Self-Flow, arXiv:2603.06507) on the
K2 backbone via extension seams and runtime forward hooks/monkeypatches —
zero modifications to ``krea2_mmdit.py`` or any base trainer are required.

Per-token conditioning is achieved with one ``register_forward_hook`` on
``model.tmlp`` (which drives every block's modulation automatically via
``tproj``, a pure elementwise broadcast) plus one runtime instance-attribute
monkeypatch of ``model.last.modulation.forward`` (the final layer's
``SimpleModulation`` cannot take a per-token vec via hooking alone — its
internal scale/shift broadcast trick only supports a token-axis size of 1 or
2, and raises inside the original forward before any hook can intervene).
See docs/superpowers/specs/2026-07-27-krea2-self-flow-design.md for the
verification behind this mechanism.

Limitations (first pass, matching the FLUX.2 port): coupling-prob decay
schedules are constant-only, patch-locality mask modes are not ported.

Internal extension point — no API stability guarantees.
"""

import argparse
import logging

import torch
from accelerate import Accelerator

from musubi_tuner.hv_train_network import setup_parser_common, read_config_from_file
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# region self-flow math helpers (ported verbatim from flux_2_train_network_self_flow.py —
# these 7 functions are architecture-agnostic pure tensor ops; only apply_per_token_mask
# differs for K2, see Task 3)


def assign_teacher_student_timesteps(timesteps_a: torch.Tensor, timesteps_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample teacher/student split: teacher = min (cleaner), student = max.

    Paper Eq. 4: masked tokens get s, unmasked get t; teacher sees the uniform
    cleaner timestep tau_min = min(t, s). See build_per_token_timestep_map for
    the per-token map construction.
    """
    t_a = timesteps_a.float()
    t_b = timesteps_b.float()
    timesteps_teacher = torch.min(t_a, t_b).to(timesteps_a.dtype)
    timesteps_student = torch.max(t_a, t_b).to(timesteps_a.dtype)
    return timesteps_teacher, timesteps_student


def reconstruct_noisy_input(latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """Flow-matching interpolation (1-t)*latents + t*noise; timesteps in [1, 1001]."""
    t = (timesteps.float() - 1.0) / 1000.0
    if latents.ndim == 5:
        t_exp = t.view(-1, 1, 1, 1, 1)
    else:
        t_exp = t.view(-1, 1, 1, 1)
    return (1 - t_exp) * latents + t_exp * noise


def apply_per_token_mask(
    noisy_input_student: torch.Tensor,
    noisy_input_teacher: torch.Tensor,
    mask_ratio: "float | torch.Tensor",
    patch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-patch-token masking (paper Eq. 4-5), K2-specific.

    Unlike FLUX.2 (patch=1, one token per latent pixel), K2 groups
    ``patch x patch`` latent pixels into one DiT token before the DiT ever
    sees it. The random mask/mismatch draw must therefore happen once per
    DiT token (shape (B, h_*w_)), not once per latent pixel — otherwise the
    mask used here (pixel-space) and the mask used for the modulation hook
    (token-space) would disagree. Each token's draw is expanded to its full
    ``patch x patch`` pixel block via repeat_interleave before selecting
    between student/teacher.

    ``noisy_input_student``/``noisy_input_teacher`` are (B, C, 1, H, W) —
    K2 always has a single-frame T=1 axis.

    Returns (masked_input, mask_tok) with mask_tok (B, h_*w_) bool,
    True = masked (this sample's patch-token takes the teacher/cleaner value).
    """
    B, C, T, H, W = noisy_input_student.shape
    assert T == 1, f"K2 expects single-frame latents, got T={T}"
    h_tok, w_tok = H // patch, W // patch
    N = h_tok * w_tok

    ratio = torch.as_tensor(mask_ratio, dtype=torch.float32, device=device)
    if ratio.ndim == 0:
        ratio = ratio.expand(B)
    mask_tok = torch.rand(B, N, device=device) < ratio.unsqueeze(1)  # (B, N)

    mask_grid = mask_tok.view(B, 1, 1, h_tok, w_tok)
    mask_pixel = mask_grid.repeat_interleave(patch, dim=3).repeat_interleave(patch, dim=4)
    mask_pixel = mask_pixel.expand(B, C, T, H, W)

    masked_input = torch.where(mask_pixel, noisy_input_teacher, noisy_input_student)
    return masked_input, mask_tok


def build_per_token_timestep_map(
    timesteps_teacher: torch.Tensor,
    timesteps_student: torch.Tensor,
    mask_flat: torch.Tensor,
    mismatch_prob: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token timestep map for dual-timestep conditioning (paper Eq. 4).

    Unmasked tokens get the student timestep. Masked tokens get the teacher
    timestep, except with probability mismatch_prob they get the student
    timestep (deliberate mismatch — experimental, 0.0 = paper behaviour).
    Returns (per_token_timesteps (B, N), mismatch_mask (B, N) bool).
    """
    B, N = mask_flat.shape
    t_student = timesteps_student.unsqueeze(1).expand(B, N)
    t_teacher = timesteps_teacher.unsqueeze(1).expand(B, N)

    if mismatch_prob <= 0.0:
        per_token_t = torch.where(mask_flat, t_teacher, t_student)
        return per_token_t, torch.zeros_like(mask_flat)

    coin = torch.rand(B, N, device=mask_flat.device)
    mismatch_mask = mask_flat & (coin < mismatch_prob)
    per_token_t = torch.where(mask_flat & ~mismatch_mask, t_teacher, t_student)
    return per_token_t, mismatch_mask


def update_ema_weights(
    ema_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    decay: float,
) -> None:
    """In-place EMA update: ema = decay * ema + (1 - decay) * current."""
    with torch.no_grad():
        for k, v in current_state.items():
            if v.is_floating_point():
                ema_state[k].lerp_(v, 1 - decay)
            else:
                ema_state[k].copy_(v)


def compute_ema_weight_drift(
    ema_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Mean L2 distance between EMA and current weights (floating-point only)."""
    with torch.no_grad():
        dists = [
            torch.linalg.vector_norm(current_state[k].float() - v.float()) for k, v in ema_state.items() if v.is_floating_point()
        ]
        if not dists:
            return torch.tensor(0.0)
        return torch.stack(dists).mean()


def compute_representation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    rep_proj: torch.nn.Module,
) -> torch.Tensor:
    """L_rep (paper Eq. 6): negative mean cosine similarity of projected student vs teacher."""
    student_proj = rep_proj(student_features)
    cos_sim = torch.nn.functional.cosine_similarity(student_proj, teacher_features, dim=-1)
    return -cos_sim.mean()


def effective_gamma(gamma: float, global_step: int, warmup_steps: int) -> float:
    """Linear warmup of the L_rep weight: 0 -> gamma over warmup_steps, then constant."""
    if warmup_steps <= 0:
        return gamma
    return gamma * min(1.0, global_step / warmup_steps)


# endregion self-flow math helpers


class Krea2SelfFlowNetworkTrainer(Krea2NetworkTrainer):
    pass


def self_flow_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return parser


def main():
    parser = setup_parser_common()
    parser = krea2_setup_parser(parser)
    parser = self_flow_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = Krea2SelfFlowNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
