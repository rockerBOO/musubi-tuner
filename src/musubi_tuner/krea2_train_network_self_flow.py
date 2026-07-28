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

from accelerate import Accelerator

from musubi_tuner.hv_train_network import setup_parser_common, read_config_from_file
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
