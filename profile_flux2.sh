#!/usr/bin/env bash
# profile_flux2.sh — run Flux2 training with PyTorch profiler
#
# Usage:
#   ./profile_flux2.sh --dataset_config my.toml --output_dir output [all normal training args]
#
# Env vars (optional overrides):
#   PROFILE_WARMUP=N        steps to skip before recording (default: 2)
#   PROFILE_STEPS=N         steps to actively record (default: 5)
#   PROFILE_OUTPUT_DIR=dir  output root dir (default: profiling)
#
# Viewing output:
#   Chrome trace — open trace_0.json at https://ui.perfetto.dev
#   Flamegraph   — flamegraph.pl cpu_stacks_0.txt > flamegraph_cpu.svg
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_DIR="${PROFILE_OUTPUT_DIR:-profiling}/$TIMESTAMP"
mkdir -p "$PROFILE_DIR"

accelerate launch \
  --num_cpu_threads_per_process 1 \
  --mixed_precision bf16 \
  src/musubi_tuner/flux_2_profiler_train_network.py \
  --profile_warmup "${PROFILE_WARMUP:-2}" \
  --profile_steps "${PROFILE_STEPS:-5}" \
  --profile_output_dir "$PROFILE_DIR" \
  "$@"

echo ""
echo "Profiling output saved to: $PROFILE_DIR"
echo "  Chrome trace : open $PROFILE_DIR/trace_0.json at https://ui.perfetto.dev"

if command -v flamegraph.pl &>/dev/null; then
  flamegraph.pl "$PROFILE_DIR/cpu_stacks_0.txt"  > "$PROFILE_DIR/flamegraph_cpu.svg"
  flamegraph.pl "$PROFILE_DIR/cuda_stacks_0.txt" > "$PROFILE_DIR/flamegraph_cuda.svg"
  echo "  CPU flamegraph : $PROFILE_DIR/flamegraph_cpu.svg"
  echo "  CUDA flamegraph: $PROFILE_DIR/flamegraph_cuda.svg"
else
  echo "  Flamegraph (install flamegraph.pl to auto-generate):"
  echo "    flamegraph.pl $PROFILE_DIR/cpu_stacks_0.txt > flamegraph_cpu.svg"
fi
