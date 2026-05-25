#!/usr/bin/env bash
# Profile run for abstract-flux2
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
  --output_dir=/mnt/900/training/sets/abstract-flux2-2026-05-25-135724-9dfa189b \
  --logging_dir=/mnt/900/training/log/abstract-flux2 \
  --log_prefix=abstract-flux2-2026-05-25-135724-9dfa189b \
  --config_file=/home/rockerboo/code/training_runs/abstract-flux2/abstract-flux2_config.toml \
  --dataset_config=/home/rockerboo/code/training_runs/abstract-flux2/dataset_config.toml \
  --output_name=abstract-flux2-2026-05-25-135724-9dfa189b \
  --metadata_title=Abstract \
  --metadata_author=rockerBOO \
  '--metadata_description=Abstract stylized images' \
  '--metadata_license=CreativeML Open RAIL-M' \
  --metadata_tags=abstract \
  --sample_prompts=/home/rockerboo/code/training_runs/abstract-flux2/abstract-flux2.txt \
  '--training_comment=Trained by: rockerBOO' \
  --log_tracker_config=/home/rockerboo/code/training_runs/abstract-flux2/abstract-flux2_log_config.toml \
  --log_tracker_name=abstract-flux2-kohya-lora \
  --dit=/mnt/500c/models/diffusion/flux2/flux-2-klein-base-4b.safetensors \
  --text_encoder=/mnt/500c/models/text_encoders/qwen_3_4b.safetensors \
  --vae=/mnt/900/vae/flux2-vae-non-diffusers.safetensors \
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
