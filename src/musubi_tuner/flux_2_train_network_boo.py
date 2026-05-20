"""BooTrainerOrchestrator — self-flow + wavelet loss combined for FLUX.2.

Uses TrainerOrchestrator to compose Flux2SelfFlowNetworkTrainer and
Flux2WaveletLossNetworkTrainer without multiple inheritance.

process_batch is owned by self-flow (dual-timestep, EMA teacher).
compute_loss wires wavelet's auxiliary loss on top of self-flow's output.

Internal extension point — no API stability guarantees.
"""

import argparse
import logging

import torch
from accelerate import Accelerator

from musubi_tuner.flux_2_train_network import flux2_setup_parser
from musubi_tuner.flux_2_train_network_self_flow import (
    Flux2SelfFlowNetworkTrainer,
    self_flow_setup_parser,
)
from musubi_tuner.flux_2_train_network_wavelet_loss import (
    Flux2WaveletLossNetworkTrainer,
    wavelet_loss_setup_parser,
)
from musubi_tuner.hv_train_network import DiTOutput, setup_parser_common, read_config_from_file
from musubi_tuner.training.orchestrator import TrainerOrchestrator

logger = logging.getLogger(__name__)


class BooTrainerOrchestrator(TrainerOrchestrator):
    """FLUX.2 trainer combining self-flow and wavelet loss via orchestration.

    Self-flow owns process_batch (dual timestep scheduling, EMA teacher).
    Wavelet loss is applied on top via compute_loss after the self-flow
    forward pass produces its DiTOutput.

    Both extensions' void hooks (on_train_start, on_post_optimizer_step, etc.)
    are called automatically by the base orchestrator.
    """

    def __init__(self) -> None:
        super().__init__()
        self._self_flow = Flux2SelfFlowNetworkTrainer()
        self._wavelet = Flux2WaveletLossNetworkTrainer()
        self.add_extension(self._self_flow)
        self.add_extension(self._wavelet)

    def process_batch(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        network,
        batch: dict[str, torch.Tensor],
        latents: torch.Tensor,
        noise: torch.Tensor,
        noise_scheduler,
        dit_dtype: torch.dtype,
        network_dtype: torch.dtype,
        vae,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Self-flow drives the forward pass; wavelet loss is added on top.

        Self-flow's process_batch is NotImplementedError until PR #913 is ported.
        Once implemented, it returns (loss, metrics) from its own compute_loss call.
        Wavelet's auxiliary loss must then be added on top — this requires either:
          a) self-flow exposes the DiTOutput so wavelet can run on it, or
          b) wavelet's compute_loss is called here with a re-derived x0.

        This stub delegates to self-flow only and marks wavelet composition as TODO.
        Fill in once self-flow's process_batch is implemented.
        """
        # Self-flow owns the primary forward pass
        loss, metrics = self._self_flow.process_batch(
            args, accelerator, transformer, network, batch, latents, noise,
            noise_scheduler, dit_dtype, network_dtype, vae, global_step,
        )

        # TODO: apply wavelet auxiliary loss on top once self-flow exposes DiTOutput.
        # Wavelet needs output.extra["noise"] and output.extra["noisy_model_input"]
        # stashed by its call_dit override; those are not available through
        # self-flow's process_batch return value today.
        # Options:
        #   1. self-flow returns DiTOutput in metrics, wavelet runs compute_loss on it
        #   2. re-run call_dit here with wavelet's call_dit to get stashed tensors
        #   3. wavelet call_dit is called from within self-flow's process_batch

        return loss, metrics


def main():
    parser = setup_parser_common()
    parser = flux2_setup_parser(parser)
    parser = self_flow_setup_parser(parser)
    parser = wavelet_loss_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    trainer = BooTrainerOrchestrator()
    trainer.train(args)


if __name__ == "__main__":
    main()
