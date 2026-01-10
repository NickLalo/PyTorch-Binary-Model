#!/usr/bin/env python3
"""
Pi inference smoke test for ExecuTorch .pte model.

What it does:
  1) Loads binary_mlp_xnnpack.pte from the current directory (or --pte_path) and times load.
  2) Runs N random inferences on binary inputs of length --input_dim.
  3) Prints one example input/output plus timing stats across all inferences.

Notes:
  - This script assumes your exported model expects a float32 tensor shaped (1, input_dim).
  - Default input_dim is 4 (for 4-bit integers 0..15).
"""

import argparse
import time
from pathlib import Path
from typing import List

start_time = time.time()
import torch
end_time = time.time()
print(f"Imported torch in {(end_time - start_time):.3f} s")

start_time = time.time()
from executorch.runtime import Runtime, Verification
end_time = time.time()
print(f"Imported executorch.runtime in {(end_time - start_time):.3f} s")


SEED = 112


def bits_to_int(bits: List[int]) -> int:
    """MSB-first binary list to integer."""
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def format_bits(bits: List[int]) -> str:
    return " ".join(str(int(b)) for b in bits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple ExecuTorch inferences on the Pi.")
    parser.add_argument("--pte_path", type=str, default="binary_mlp_xnnpack.pte", help="Path to .pte file.")
    parser.add_argument("--input_dim", type=int, default=4, help="Number of input bits (default: 4).")
    parser.add_argument("--num_runs", type=int, default=1000, help="Number of random inferences to run.")
    args = parser.parse_args()

    torch.manual_seed(SEED)

    pte_path = Path(args.pte_path).expanduser().resolve()
    if not pte_path.exists():
        raise FileNotFoundError(f"Could not find .pte file at: {pte_path}")
    if args.num_runs < 1:
        raise ValueError("--num_runs must be >= 1")

    print(f"Loading ExecuTorch program: {pte_path}")

    load_start = time.perf_counter()
    rt = Runtime.get()
    program = rt.load_program(pte_path, verification=Verification.Minimal)
    forward = program.load_method("forward")
    load_ms = (time.perf_counter() - load_start) * 1000.0

    times_ms = []
    num_correct = 0
    example = None

    for _ in range(args.num_runs):
        bits_i64 = torch.randint(0, 2, (args.input_dim,), dtype=torch.int64)
        bits_list = bits_i64.tolist()
        expected = bits_to_int(bits_list)

        x = bits_i64.to(dtype=torch.float32).unsqueeze(0).contiguous()

        infer_start = time.perf_counter()
        out = forward.execute((x,))[0]
        infer_ms = (time.perf_counter() - infer_start) * 1000.0
        times_ms.append(infer_ms)

        if not isinstance(out, torch.Tensor):
            out = torch.tensor(out)

        logits = out.to(dtype=torch.float32).cpu()
        pred = int(torch.argmax(logits, dim=1).item())
        num_correct += int(pred == expected)

        if example is None:
            example = (bits_list, expected, logits, pred)

    if example is not None:
        bits_list, expected, logits, pred = example
        print(f"Input bits:      {format_bits(bits_list)}")
        print(f"Correct integer: {expected}")
        print(f"Logits:          {logits.numpy().tolist()}")
        print(f"Predicted class: {pred}")
        print(f"Correct:         {pred == expected}")

    times_ms.sort()
    total_ms = sum(times_ms)
    avg_ms = total_ms / len(times_ms)
    min_ms = times_ms[0]
    max_ms = times_ms[-1]
    p50_ms = times_ms[int(0.50 * (len(times_ms) - 1))]
    p95_ms = times_ms[int(0.95 * (len(times_ms) - 1))]
    accuracy = num_correct / len(times_ms) if times_ms else 0.0

    print(
        "Timing:          "
        f"load={load_ms:.3f} ms | "
        f"runs={len(times_ms)} | "
        f"total={total_ms:.3f} ms | "
        f"avg={avg_ms:.3f} ms | "
        f"p50={p50_ms:.3f} ms | "
        f"p95={p95_ms:.3f} ms | "
        f"min={min_ms:.3f} ms | "
        f"max={max_ms:.3f} ms"
    )
    print(f"Accuracy:        {accuracy:.3%} ({num_correct}/{len(times_ms)})")

    return


if __name__ == "__main__":
    main()
