#!/usr/bin/env bash
# profile_flux2.sh — run Flux2 training with PyTorch profiler
#
# Usage (drop-in for your normal training command):
#   ./profile_flux2.sh \
#     --config_file=/path/to/config.toml \
#     --dataset_config=/path/to/dataset_config.toml \
#     --dit=... --text_encoder=... --vae=... [all other normal args]
#
# Env vars (optional overrides):
#   PROFILE_WARMUP=N        steps to skip before recording (default: 2)
#   PROFILE_STEPS=N         steps to actively record (default: 5)
#   PROFILE_OUTPUT_DIR=dir  output root dir (default: profiling)
#   MIXED_PRECISION=value   accelerate mixed precision (default: bf16)
#
# Viewing output:
#   Chrome trace — open trace_N.json at https://ui.perfetto.dev
#   Flamegraph   — flamegraph.pl cpu_stacks_N.txt > flamegraph_cpu.svg
#   (N = PROFILE_WARMUP + PROFILE_STEPS, default: 7)
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_DIR="${PROFILE_OUTPUT_DIR:-profiling}/$TIMESTAMP"
PROFILE_SUFFIX=$(( ${PROFILE_WARMUP:-2} + ${PROFILE_STEPS:-5} ))
mkdir -p "$PROFILE_DIR"

uv run --no-sync accelerate launch \
  --num_cpu_threads_per_process 1 \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  flux_2_profiler_train_network.py \
  --profile_warmup "${PROFILE_WARMUP:-2}" \
  --profile_steps "${PROFILE_STEPS:-5}" \
  --profile_output_dir "$PROFILE_DIR" \
  "$@"

echo ""
echo "Profiling output saved to: $PROFILE_DIR"
echo "  Chrome trace : open $PROFILE_DIR/trace_${PROFILE_SUFFIX}.json at https://ui.perfetto.dev"

if command -v flamegraph.pl &>/dev/null; then
  if [[ -f "$PROFILE_DIR/cpu_stacks_${PROFILE_SUFFIX}.txt" ]]; then
    flamegraph.pl "$PROFILE_DIR/cpu_stacks_${PROFILE_SUFFIX}.txt"  > "$PROFILE_DIR/flamegraph_cpu.svg"
    flamegraph.pl "$PROFILE_DIR/cuda_stacks_${PROFILE_SUFFIX}.txt" > "$PROFILE_DIR/flamegraph_cuda.svg"
    echo "  CPU flamegraph : $PROFILE_DIR/flamegraph_cpu.svg"
    echo "  CUDA flamegraph: $PROFILE_DIR/flamegraph_cuda.svg"
  else
    echo "  Stack files not found — profiling window may not have completed (need >= $PROFILE_SUFFIX steps)."
  fi
else
  echo "  Flamegraph (install flamegraph.pl to auto-generate):"
  echo "    flamegraph.pl $PROFILE_DIR/cpu_stacks_${PROFILE_SUFFIX}.txt > flamegraph_cpu.svg"
fi
