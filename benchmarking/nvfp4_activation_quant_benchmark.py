"""Isolated benchmark: quantize_nvfp4_activation vs ConvRot INT8's fused quantize-and-rotate
step vs a bare scaled_mm call, at Krea2's real per-block Linear shapes.

Run: uv run --no-sync python benchmarking/nvfp4_activation_quant_benchmark.py
Requires: CUDA, PyTorch 2.10+ (torch.nn.functional.scaled_mm), a Blackwell GPU for the scaled_mm
and full nvfp4_scaled_mm_linear measurements (quantize_nvfp4_activation itself is pure tensor
math and will run on any CUDA GPU, but comparing it against a real scaled_mm call needs Blackwell).
"""

import time

import torch

from musubi_tuner.modules.nvfp4_utils import (
    nvfp4_scaled_mm_available,
    nvfp4_scaled_mm_linear,
    quantize_nvfp4_activation,
)


def _bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / iters * 1000
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return elapsed_ms, peak_mb


def bench_quantize_nvfp4_activation(m: int, k: int):
    device = torch.device("cuda")
    x = (torch.randn(m, k, device=device) * 0.5).to(torch.bfloat16)

    def run():
        quantize_nvfp4_activation(x.float())

    return _bench(run)


def bench_quantize_nvfp4_activation_stochastic(m: int, k: int):
    from musubi_tuner.modules.nvfp4_utils import quantize_nvfp4_activation_stochastic

    device = torch.device("cuda")
    x = (torch.randn(m, k, device=device) * 0.5).to(torch.bfloat16)

    def run():
        quantize_nvfp4_activation_stochastic(x.float())

    return _bench(run)


def bench_convrot_int8_quantize_rotate(m: int, k: int):
    from musubi_tuner.modules.convrot_int8_kernels import _build_hadamard, _rotate_activation, triton_quantize_rowwise
    from musubi_tuner.modules.convrot_int8_utils import CONVROT_GROUPSIZE

    device = torch.device("cuda")
    x = (torch.randn(m, k, device=device) * 0.5).to(torch.bfloat16)
    h = _build_hadamard(CONVROT_GROUPSIZE, device=device, dtype=x.dtype)

    def run():
        x_rotated = _rotate_activation(x, h, CONVROT_GROUPSIZE)
        triton_quantize_rowwise(x_rotated)

    return _bench(run)


def bench_bare_scaled_mm(n: int, k: int, m: int):
    from musubi_tuner.modules.nvfp4_utils import _quantize_nvfp4_2d

    device = torch.device("cuda")
    x = (torch.randn(m, k, device=device) * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device=device) * 0.02).to(torch.bfloat16)
    x_packed, x_block_scale, x_scale, _ = _quantize_nvfp4_2d(x.float())
    w_packed, w_block_scale, w_scale, _ = _quantize_nvfp4_2d(w.float())

    from torch.nn.functional import ScalingType, SwizzleType

    def run():
        torch.nn.functional.scaled_mm(
            x_packed.view(torch.float4_e2m1fn_x2),
            w_packed.view(torch.float4_e2m1fn_x2).t(),
            scale_a=[x_block_scale.view(-1), x_scale],
            scale_b=[w_block_scale.view(-1), w_scale],
            bias=None,
            output_dtype=torch.bfloat16,
            scale_recipe_a=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            scale_recipe_b=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            swizzle_a=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
            swizzle_b=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
        )

    return _bench(run)


def bench_full_nvfp4_scaled_mm_linear(n: int, k: int, m: int):
    from musubi_tuner.modules.nvfp4_utils import _quantize_nvfp4_2d

    device = torch.device("cuda")
    x = (torch.randn(m, k, device=device) * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device=device) * 0.02).to(torch.bfloat16)
    w_packed, w_block_scale, w_scale, _ = _quantize_nvfp4_2d(w.float())

    def run():
        nvfp4_scaled_mm_linear(x, w_packed, w_block_scale, w_scale, None, n)

    return _bench(run)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    if not nvfp4_scaled_mm_available():
        raise SystemExit("PyTorch 2.10+ with scaled_mm/float4_e2m1fn_x2 required")

    shapes = [
        ("mlp.gate (largest)", 24576, 6144),
        ("attn.wk (N<K, smallest)", 1536, 6144),
    ]
    m = 4096  # matches the reviewer's original isolated measurement

    print(f"{'shape':<28} {'op':<32} {'ms/call':>10} {'peak MiB':>10}")
    for label, n, k in shapes:
        ms, mb = bench_quantize_nvfp4_activation(m, k)
        print(f"{label:<28} {'quantize_nvfp4_activation':<32} {ms:>10.2f} {mb:>10.0f}")

        ms, mb = bench_quantize_nvfp4_activation_stochastic(m, k)
        print(f"{label:<28} {'quantize_nvfp4_activation_stochastic (grad_out path)':<32} {ms:>10.2f} {mb:>10.0f}")

        try:
            ms, mb = bench_convrot_int8_quantize_rotate(m, k)
            print(f"{label:<28} {'convrot_int8 quantize+rotate':<32} {ms:>10.2f} {mb:>10.0f}")
        except (ImportError, AttributeError) as e:
            print(f"{label:<28} {'convrot_int8 quantize+rotate':<32} SKIPPED ({e})")

        ms, mb = bench_bare_scaled_mm(n, k, m)
        print(f"{label:<28} {'bare scaled_mm':<32} {ms:>10.2f} {mb:>10.0f}")

        ms, mb = bench_full_nvfp4_scaled_mm_linear(n, k, m)
        print(f"{label:<28} {'full nvfp4_scaled_mm_linear':<32} {ms:>10.2f} {mb:>10.0f}")
        print()


if __name__ == "__main__":
    main()
