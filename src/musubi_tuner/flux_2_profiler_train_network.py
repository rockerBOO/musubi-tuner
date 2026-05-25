"""Profiling trainer for FLUX.2 — wraps the normal training loop with PyTorch profiler.

Run this script instead of flux_2_train_network.py when you want profiling output.
Use profile_flux2.sh for a convenient launcher.

Output (in --profile_output_dir / $PROFILE_OUTPUT_DIR):
  trace_N.json       — Chrome/Perfetto trace (CPU+CUDA timeline)
  cpu_stacks_N.txt   — flamegraph.pl input for CPU overhead
  cuda_stacks_N.txt  — flamegraph.pl input for CUDA kernels

Key averages table prints to stdout when the profiling window closes.
"""

import argparse
import logging
import os
import time
from typing import Optional

import torch
from torch.profiler import ProfilerActivity
from accelerate import Accelerator

from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer, flux2_setup_parser
from musubi_tuner.hv_train_network import setup_parser_common, read_config_from_file

logger = logging.getLogger(__name__)


def add_profiler_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("profiler")
    group.add_argument(
        "--profile_warmup",
        type=int,
        default=2,
        help="steps to skip before recording (default: 2)",
    )
    group.add_argument(
        "--profile_steps",
        type=int,
        default=5,
        help="steps to actively record (default: 5)",
    )
    group.add_argument(
        "--profile_output_dir",
        type=str,
        default="profiling",
        help="directory for profiling output (default: profiling)",
    )
    return parser


class Flux2ProfilerNetworkTrainer(Flux2NetworkTrainer):
    """Flux2 trainer with PyTorch profiler instrumentation.

    Adds per-step timing (forward / backward / optimizer) logged to
    wandb/tensorboard, plus a full PyTorch profiler trace covering
    --profile_warmup warmup steps and --profile_steps active steps.
    """

    def __init__(self) -> None:
        super().__init__()
        self.profiler: Optional[torch.profiler.profile] = None
        self.profile_output_dir: str = "profiling"
        self._t_step_start: float = 0.0
        self._t_forward_end: float = 0.0
        self._t_backward_start: float = 0.0
        self._t_backward_end: float = 0.0
        self._t_optimizer_end: float = 0.0

    # region profiler lifecycle

    def on_train_start(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        optimizer,
    ) -> None:
        super().on_train_start(args, accelerator, network, transformer, optimizer)

        self.profile_output_dir = args.profile_output_dir
        os.makedirs(self.profile_output_dir, exist_ok=True)

        output_dir = self.profile_output_dir

        def trace_handler(p: torch.profiler.profile) -> None:
            step = p.step_num
            p.export_chrome_trace(os.path.join(output_dir, f"trace_{step}.json"))
            p.export_stacks(
                os.path.join(output_dir, f"cpu_stacks_{step}.txt"),
                metric="self_cpu_time_total",
            )
            p.export_stacks(
                os.path.join(output_dir, f"cuda_stacks_{step}.txt"),
                metric="self_cuda_time_total",
            )
            logger.info(f"Profiling trace saved to {output_dir}/trace_{step}.json")
            print(
                p.key_averages().table(sort_by="cuda_time_total", row_limit=20)
            )

        self.profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=args.profile_warmup,
                active=args.profile_steps,
                repeat=1,
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )
        self.profiler.__enter__()
        logger.info(
            f"Profiler started — warmup={args.profile_warmup} steps, "
            f"active={args.profile_steps} steps, output={output_dir}"
        )

    # endregion

    # region per-step timing hooks

    def process_batch(self, args, accelerator, transformer, network, batch, latents, noise,
                      noise_scheduler, dit_dtype, network_dtype, vae, global_step):
        self._t_step_start = time.perf_counter()
        result = super().process_batch(
            args, accelerator, transformer, network, batch, latents, noise,
            noise_scheduler, dit_dtype, network_dtype, vae, global_step,
        )
        self._t_forward_end = time.perf_counter()
        return result

    def on_before_backward(self, loss: torch.Tensor) -> None:
        self._t_backward_start = time.perf_counter()

    def on_after_backward(self) -> None:
        self._t_backward_end = time.perf_counter()

    def on_post_optimizer_step(self, args, accelerator, network, transformer, sync_gradients, global_step) -> None:
        self._t_optimizer_end = time.perf_counter()
        super().on_post_optimizer_step(args, accelerator, network, transformer, sync_gradients, global_step)

    # endregion

    def _compute_timing_logs(self) -> dict:
        return {
            "profile/forward_ms":   (self._t_forward_end   - self._t_step_start)     * 1000,
            "profile/backward_ms":  (self._t_backward_end  - self._t_backward_start) * 1000,
            "profile/optimizer_ms": (self._t_optimizer_end - self._t_backward_end)   * 1000,
            "profile/step_ms":      (self._t_optimizer_end - self._t_step_start)     * 1000,
        }

    def extra_step_logs(self, args: argparse.Namespace, logs: dict) -> dict:
        if self.profiler is not None:
            self.profiler.step()

        result = super().extra_step_logs(args, logs)
        result.update(self._compute_timing_logs())
        return result


def main():
    parser = setup_parser_common()
    parser = flux2_setup_parser(parser)
    parser = add_profiler_args(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None
    if args.vae_dtype is None:
        args.vae_dtype = "float32"

    trainer = Flux2ProfilerNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
