"""Wavelet-loss training entry point for FLUX.2.

Augments the standard FLUX.2 training loss with a frequency-domain component
via the wavelet-loss package (``pip install -e /path/to/wavelet-loss`` or add
to pyproject.toml as a path dependency).

The wavelet loss operates on *estimated clean latents* (x0) derived from the
model's velocity prediction using the flow-matching identity:

    x0_pred   = noisy_model_input - sigma * pred
    x0_target = noisy_model_input - sigma * noise   (= latents exactly)

This gives the wavelet transform meaningful frequency structure to penalise
rather than operating on the velocity residual space.

Usage (extends normal FLUX.2 training command):

    accelerate launch flux_2_train_network_wavelet_loss.py \\
        --wavelet_loss \\
        --wavelet_loss_alpha 0.1 \\
        --wavelet_loss_transform swt \\
        --wavelet_loss_level 2 \\
        <...normal FLUX.2 training args...>

Internal extension point — no API stability guarantees. Subclasses live in
this repo; if you fork, expect breakage on updates.
"""

import argparse
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer, flux2_setup_parser
from musubi_tuner.hv_train_network import (
    DiTOutput,
    setup_parser_common,
    read_config_from_file,
)
from musubi_tuner.training.timesteps import compute_loss_weighting_for_sd3, get_sigmas

try:
    from wavelet_loss import WaveletLoss
except ImportError:
    WaveletLoss = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Flux2WaveletLossNetworkTrainer(Flux2NetworkTrainer):
    """FLUX.2 + wavelet-domain auxiliary loss.

    Owned state:
    - ``self.wavelet_loss``: ``WaveletLoss`` module, constructed in
      ``on_train_start`` when ``--wavelet_loss`` is set. Holds the wavelet
      filters as registered buffers so they move to the correct device
      automatically.
    """

    def __init__(self) -> None:
        super().__init__()
        self.wavelet_loss: Optional["WaveletLoss"] = None  # type: ignore[type-arg]

    # region argument validation

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        super().handle_model_specific_args(args)
        if args.wavelet_loss and WaveletLoss is None:
            raise ImportError(
                "wavelet-loss package is not installed. "
                "Install it with: pip install -e /path/to/wavelet-loss"
            )

    # endregion

    # region extension seam overrides

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        **kwargs,
    ) -> DiTOutput:
        """Delegates to parent and stashes noise / noisy_model_input for compute_loss.

        Both tensors are already available as arguments here; putting them in
        ``DiTOutput.extra`` avoids changing the ``compute_loss`` signature while
        keeping the wavelet computation self-contained in that hook.
        """
        output = super().call_dit(
            args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
        )
        output.extra["noise"] = noise
        output.extra["noisy_model_input"] = noisy_model_input
        return output

    def on_train_start(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        optimizer,
    ) -> None:
        """Construct and move the WaveletLoss module to the training device."""
        if not args.wavelet_loss:
            return

        assert WaveletLoss is not None, "wavelet-loss package not installed"
        device = accelerator.device

        self.wavelet_loss = WaveletLoss(
            transform_type=args.wavelet_loss_transform,
            wavelet=args.wavelet_loss_wavelet,
            level=args.wavelet_loss_level,
            band_weights=args.wavelet_loss_band_weights,
            band_level_weights=args.wavelet_loss_band_level_weights,
            quaternion_component_weights=args.wavelet_loss_quaternion_component_weights,
            ll_level_threshold=args.wavelet_loss_ll_level_threshold,
            metrics=args.wavelet_loss_metrics,
            normalize_bands=args.wavelet_loss_normalize_bands,
            timestep_intensity=args.wavelet_loss_timestep_intensity,
            use_snr_aware_huber=args.wavelet_loss_use_snr_aware_huber,
            snr_huber_cmin=args.wavelet_loss_snr_huber_cmin,
            snr_huber_cmax=args.wavelet_loss_snr_huber_cmax,
            snr_huber_gamma=args.wavelet_loss_snr_huber_gamma,
            snr_huber_alpha=args.wavelet_loss_snr_huber_alpha,
            min_snr_beta=args.wavelet_loss_min_snr_beta,
            device=device,
        )
        assert self.wavelet_loss is not None
        self.wavelet_loss.to(device)

        logger.info("Wavelet loss enabled:")
        logger.info(f"\tTransform: {args.wavelet_loss_transform}")
        logger.info(f"\tWavelet:   {args.wavelet_loss_wavelet}")
        logger.info(f"\tLevel:     {args.wavelet_loss_level}")
        logger.info(f"\tAlpha:     {args.wavelet_loss_alpha}")
        if args.wavelet_loss_primary:
            logger.info("\tMode:      primary (replaces MSE loss)")
        if args.wavelet_loss_band_weights:
            logger.info(f"\tBand weights: {args.wavelet_loss_band_weights}")
        if args.wavelet_loss_use_snr_aware_huber:
            logger.info("\tSNR-aware Huber enabled")
            logger.info(f"\t\tcmin={args.wavelet_loss_snr_huber_cmin}, cmax={args.wavelet_loss_snr_huber_cmax}")

    def compute_loss(
        self,
        args: argparse.Namespace,
        output: DiTOutput,
        timesteps: torch.Tensor,
        noise_scheduler,
        dit_dtype: torch.dtype,
        network_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Weighted MSE + optional wavelet auxiliary loss.

        When ``--wavelet_loss`` is active the method:

        1. Derives estimated clean latents (x0) from the velocity prediction
           using the flow-matching identity.
        2. Runs ``WaveletLoss.forward`` on the x0 estimates.
        3. Combines per-level wavelet losses with the base MSE:
           - For each wavelet level, downsamples the MSE to the level's
             spatial resolution, adds ``alpha * wav_loss``, then upsamples
             back. All levels are averaged.
           - With ``--wavelet_loss_primary`` the roles are swapped: wavelet
             loss is the main signal and MSE is scaled by ``alpha``.

        ``loss_metrics`` carries all ``wavelet_loss/*`` keys for per-step
        logging alongside the normal loss/current and loss/average keys.
        """
        weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, noise_scheduler, timesteps, timesteps.device, dit_dtype)
        mse_loss = F.mse_loss(output.pred.to(network_dtype), output.target, reduction="none")
        if weighting is not None:
            mse_loss = mse_loss * weighting

        if not args.wavelet_loss or self.wavelet_loss is None:
            return mse_loss.mean(), {}

        # --- wavelet path ---

        noise = output.extra["noise"]
        noisy_model_input = output.extra["noisy_model_input"]

        # flow-matching x0 recovery:
        #   noisy = (1-sigma)*latents + sigma*noise
        #   pred  ≈ noise - latents  (velocity)
        #   x0    = noisy - sigma * pred  =  latents  (exactly when pred is perfect)
        sigmas = get_sigmas(noise_scheduler, timesteps, noisy_model_input.device, n_dim=output.pred.ndim, dtype=output.pred.dtype)
        x0_pred = noisy_model_input - sigmas * output.pred.to(noisy_model_input.dtype)
        x0_target = noisy_model_input - sigmas * noise.to(noisy_model_input.dtype)

        # update loss function to match the current huber_c / loss_type
        # (done per-step so it picks up any scheduled changes)
        loss_type = args.wavelet_loss_type if args.wavelet_loss_type is not None else args.loss_type

        def _loss_fn(input: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
            if loss_type in ("l1", "mae"):
                return F.l1_loss(input, target, reduction=reduction)
            if loss_type in ("huber", "smooth_l1"):
                return F.smooth_l1_loss(input, target, reduction=reduction)
            return F.mse_loss(input, target, reduction=reduction)

        self.wavelet_loss.set_loss_fn(_loss_fn)

        wav_losses, wav_metrics = self.wavelet_loss(
            x0_pred.float(),
            x0_target.float(),
            timesteps,
        )
        loss_metrics = {f"wavelet_loss/{k}": v for k, v in wav_metrics.items()}

        # Combine each wavelet level with the base MSE at matching spatial resolution,
        # then upsample back and average across levels.
        combined_levels = []
        for wav_loss in wav_losses:
            target_hw = wav_loss.shape[-2:]
            # downsample MSE loss map to wavelet level resolution
            down = F.adaptive_avg_pool2d(mse_loss, target_hw)
            if args.wavelet_loss_primary:
                # wavelet is primary; MSE is auxiliary scaled by alpha
                combined = wav_loss + args.wavelet_loss_alpha * down
            else:
                combined = down + args.wavelet_loss_alpha * wav_loss
            # restore original spatial size
            up = F.interpolate(combined, size=mse_loss.shape[-2:], mode="bilinear", align_corners=False)
            combined_levels.append(up)

        loss = torch.stack(combined_levels).mean(dim=0)
        return loss.mean(), loss_metrics

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        """Embed wavelet-loss configuration into the saved safetensors metadata."""
        if not args.wavelet_loss:
            return {}
        import json

        return {
            "ss_wavelet_loss": True,
            "ss_wavelet_loss_alpha": args.wavelet_loss_alpha,
            "ss_wavelet_loss_primary": args.wavelet_loss_primary,
            "ss_wavelet_loss_type": args.wavelet_loss_type,
            "ss_wavelet_loss_transform": args.wavelet_loss_transform,
            "ss_wavelet_loss_wavelet": args.wavelet_loss_wavelet,
            "ss_wavelet_loss_level": args.wavelet_loss_level,
            "ss_wavelet_loss_band_weights": json.dumps(args.wavelet_loss_band_weights) if args.wavelet_loss_band_weights else None,
            "ss_wavelet_loss_band_level_weights": json.dumps(args.wavelet_loss_band_level_weights) if args.wavelet_loss_band_level_weights else None,
            "ss_wavelet_loss_quaternion_component_weights": json.dumps(args.wavelet_loss_quaternion_component_weights) if args.wavelet_loss_quaternion_component_weights else None,
            "ss_wavelet_loss_ll_level_threshold": args.wavelet_loss_ll_level_threshold,
        }

    # endregion


def _parse_band_weights(weights_str: Optional[str]) -> Optional[dict[str, float]]:
    """Parse ``ll=0.1,lh=0.01,hl=0.01,hh=0.05`` or JSON dict string."""
    if weights_str is None:
        return None
    import ast
    import json as _json

    if weights_str.strip().startswith("{"):
        try:
            return ast.literal_eval(weights_str)
        except (ValueError, SyntaxError):
            return _json.loads(weights_str.replace("'", '"'))

    result = {}
    for pair in weights_str.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k.strip()] = float(v.strip())
    return result


def wavelet_loss_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Wavelet-loss-specific CLI arguments."""
    parser.add_argument("--wavelet_loss", action="store_true", help="Enable wavelet auxiliary loss. Default: False")
    parser.add_argument(
        "--wavelet_loss_primary",
        action="store_true",
        help="Use wavelet loss as the primary objective; MSE becomes the auxiliary scaled by --wavelet_loss_alpha.",
    )
    parser.add_argument("--wavelet_loss_alpha", type=float, default=0.1, help="Wavelet loss weight. Default: 0.1")
    parser.add_argument(
        "--wavelet_loss_type",
        default=None,
        help="Loss function for wavelet bands: l1, l2, huber, smooth_l1. Defaults to --loss_type.",
    )
    parser.add_argument(
        "--wavelet_loss_transform",
        default="swt",
        choices=["dwt", "swt", "qwt"],
        help="Wavelet transform: dwt (discrete), swt (stationary), qwt (quaternion). Default: swt",
    )
    parser.add_argument("--wavelet_loss_wavelet", default="sym7", help="Wavelet family (e.g. sym7, db4). Default: sym7")
    parser.add_argument(
        "--wavelet_loss_level",
        type=int,
        default=1,
        help="Decomposition levels. Level 1 captures coarse structure; higher levels add detail. Default: 1",
    )
    parser.add_argument(
        "--wavelet_loss_band_weights",
        type=_parse_band_weights,
        default=None,
        help="Per-band weights as ll=0.1,lh=0.01,hl=0.01,hh=0.05 or JSON dict. Default: library defaults.",
    )
    parser.add_argument(
        "--wavelet_loss_band_level_weights",
        type=_parse_band_weights,
        default=None,
        help="Per-band-per-level weights as ll1=0.1,lh1=0.01,hh2=0.05 etc. Overrides --wavelet_loss_band_weights.",
    )
    parser.add_argument(
        "--wavelet_loss_quaternion_component_weights",
        type=_parse_band_weights,
        default=None,
        help="QWT component weights as r=1.0,i=0.7,j=0.7,k=0.5. Only used with --wavelet_loss_transform qwt.",
    )
    parser.add_argument(
        "--wavelet_loss_ll_level_threshold",
        type=int,
        default=None,
        help="Level at which to include LL (low-frequency) band. -1 = last level only. Default: None (use all).",
    )
    parser.add_argument(
        "--wavelet_loss_normalize_bands",
        action="store_true",
        default=None,
        help="Normalise each wavelet band before computing the loss.",
    )
    parser.add_argument(
        "--wavelet_loss_metrics",
        action="store_true",
        help="Log detailed per-band wavelet metrics each step (adds overhead). Default: False",
    )
    parser.add_argument(
        "--wavelet_loss_timestep_intensity",
        type=float,
        default=0.5,
        help="Timestep weighting intensity for smooth_timestep_weight. Default: 0.5",
    )
    # SNR-aware Huber (UltraFlux)
    parser.add_argument(
        "--wavelet_loss_use_snr_aware_huber",
        action="store_true",
        help="Use SNR-aware Huber loss inside wavelet bands (UltraFlux variant). Overrides --wavelet_loss_type.",
    )
    parser.add_argument("--wavelet_loss_snr_huber_cmin", type=float, default=0.2, help="SNR-aware Huber min threshold. Default: 0.2")
    parser.add_argument("--wavelet_loss_snr_huber_cmax", type=float, default=1.0, help="SNR-aware Huber max threshold. Default: 1.0")
    parser.add_argument("--wavelet_loss_snr_huber_gamma", type=float, default=5.0, help="SNR-aware Huber SNR clamp value. Default: 5.0")
    parser.add_argument(
        "--wavelet_loss_snr_huber_alpha", type=float, default=0.5, help="SNR-aware Huber transition smoothness. Default: 0.5"
    )
    parser.add_argument(
        "--wavelet_loss_min_snr_beta",
        type=float,
        default=0.0,
        help="Min-SNR weighting exponent for timestep rebalancing. 0 disables. Default: 0.0",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = flux2_setup_parser(parser)
    parser = wavelet_loss_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    trainer = Flux2WaveletLossNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
