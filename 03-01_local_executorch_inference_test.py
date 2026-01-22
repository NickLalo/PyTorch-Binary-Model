#!/usr/bin/env python3
"""
03-01_local_executorch_inference_test.py

Load an ExecuTorch .pte program (exported from the toy binary classifier) and run
per-sample inference timing tests on random binary inputs.

Behavior mirrors 03_local_pytorch_inference_test.py:
  - Resolve .pte path (via --pte_path, else binary_mlp_xnnpack.pte in cwd)
  - Generate random 0/1 bit vectors of length --input_dim
  - Compute ground-truth integer from bits (MSB first)
  - Run ExecuTorch forward() and time each run
  - Print a line per run + summary accuracy and timing stats

Notes:
  - If your exported program has a static batch size of 1 (common), this script
    runs one sample at a time with shape (1, input_dim).
  - Default input_dim is 4 to match the dataset generator.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional

import torch

from executorch.runtime import Runtime, Verification


def find_pte_path(explicit_path: Optional[str] = None) -> Path:
    """
    Resolve the .pte to load. Preference:
    1) --pte_path if provided
    2) binary_mlp_xnnpack.pte in cwd
    """
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"ExecuTorch .pte not found: {path}")
        return path

    cwd = Path.cwd()
    default_path = (cwd / "binary_mlp_xnnpack.pte").resolve()
    if not default_path.exists():
        raise FileNotFoundError(
            "No .pte found. Provide --pte_path or place binary_mlp_xnnpack.pte in the project root."
        )
    return default_path


def bits_to_int(bits: List[int]) -> int:
    """MSB-first binary list to integer."""
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def format_bits(bits: List[int]) -> str:
    return " ".join(str(int(b)) for b in bits)


def run_inference_tests(forward, input_dim: int, num_runs: int) -> None:
    start_time = time.time()
    device = torch.device("cpu")

    correct_count = 0
    durations_ms: List[float] = []

    for i in range(num_runs):
        bits = torch.randint(0, 2, (input_dim,), dtype=torch.int64)
        expected = bits_to_int(bits.tolist())

        x = bits.to(dtype=torch.float32, device=device).unsqueeze(0).contiguous()  # (1, input_dim)

        start = time.perf_counter()
        out = forward.execute((x,))[0]  # expected shape: (1, num_classes)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        pred = int(torch.argmax(out, dim=1).item())

        correct = pred == expected
        correct_count += int(correct)
        durations_ms.append(elapsed_ms)

        print(
            f"{i + 1:7,d}  |  input:{format_bits(bits.tolist())}  |  pred->{pred:<2d} "
            f"   {expected:<2d}<-ground_truth  |  correct:{correct}  |  {elapsed_ms:.3f} ms"
        )

    total = num_runs
    pct = (correct_count / total * 100.0) if total else 0.0

    mean_ms = (sum(durations_ms) / total) if total else 0.0
    if total > 1:
        mean_sq = sum((d - mean_ms) ** 2 for d in durations_ms) / total
        std_ms = mean_sq ** 0.5
    else:
        std_ms = 0.0

    max_ms = max(durations_ms) if durations_ms else 0.0

    print("\nSummary:")
    print(f"\tCorrect: {correct_count}/{total} ({pct:.2f}%)")
    print(f"\tTiming: mean={mean_ms:.3f} ms | std={std_ms:.3f} ms | max={max_ms:.3f} ms")
    end_time = time.time()
    time_hours = int((end_time - start_time) // 3600)
    time_minutes = int(((end_time - start_time) % 3600) // 60)
    time_seconds = ((end_time - start_time) % 3600) % 60
    print(f"\tTotal time for {num_runs} inferences (HH:MM:SS.ss): {time_hours:02d}:{time_minutes:02d}:{time_seconds:05.2f}")
    return


def main() -> None:
    print("Running ExecuTorch inference tests on binary classifier (.pte)...")

    parser = argparse.ArgumentParser(description="Run random inference timing tests with an ExecuTorch .pte.")
    parser.add_argument("--pte_path", type=str, default=None, help="Path to ExecuTorch .pte program.")
    parser.add_argument("--input_dim", type=int, default=4, help="Number of input bits (features).")
    parser.add_argument("--runs", type=int, default=10_000, help="Number of random inputs to test.")
    parser.add_argument(
        "--verification",
        type=str,
        default="minimal",
        choices=["none", "minimal"],
        help="ExecuTorch program verification level.",
    )
    args = parser.parse_args()

    pte_path = find_pte_path(args.pte_path)
    print(f"Loading ExecuTorch program: {pte_path}")

    verification = Verification.Minimal if args.verification == "minimal" else Verification.None_

    rt = Runtime.get()
    program = rt.load_program(pte_path, verification=verification)
    forward = program.load_method("forward")

    run_inference_tests(forward, input_dim=args.input_dim, num_runs=args.runs)
    return


if __name__ == "__main__":
    main()
