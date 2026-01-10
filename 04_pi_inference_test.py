#!/usr/bin/env python3
"""
Pi inference smoke test for ExecuTorch .pte model.

What it does:
  1) Loads binary_mlp_xnnpack.pte from the current directory (or --pte_path) and times load.
  2) Runs N random inferences on binary inputs of length --input_dim.
  3) Prints one example input/output plus timing stats across all inferences.
  4) Samples system-wide RAM usage from /proc/meminfo every N runs (default: 100) and prints stats.

Notes:
  - This script assumes your exported model expects a float32 tensor shaped (1, input_dim).
  - Default input_dim is 4 (for 4-bit integers 0..15).
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

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


def _read_meminfo_kb() -> Dict[str, int]:
    """
    Read system-wide memory info from /proc/meminfo.

    Returns values in kB. Keys included when available:
      - MemTotal
      - MemAvailable
      - MemFree
      - Buffers
      - Cached
    """
    wanted = {"MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached"}
    out: Dict[str, int] = {}

    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            key = key.strip()
            if key in wanted:
                parts = rest.strip().split()
                if parts and parts[0].isdigit():
                    out[key] = int(parts[0])

    return out


def _kb_to_mib(kb: int) -> float:
    return kb / 1024.0


def _compute_system_mem_metrics(meminfo_kb: Dict[str, int]) -> Dict[str, int]:
    """
    Compute a few derived metrics in kB:
      - used_kb: MemTotal - MemAvailable (best practical "used" approximation on Linux)
      - avail_kb: MemAvailable
      - free_kb: MemFree
      - cached_kb: Cached
      - buffers_kb: Buffers
      - total_kb: MemTotal
    Missing keys default to 0.
    """
    total_kb = int(meminfo_kb.get("MemTotal", 0))
    avail_kb = int(meminfo_kb.get("MemAvailable", 0))
    free_kb = int(meminfo_kb.get("MemFree", 0))
    cached_kb = int(meminfo_kb.get("Cached", 0))
    buffers_kb = int(meminfo_kb.get("Buffers", 0))

    used_kb = 0
    if total_kb > 0 and avail_kb > 0:
        used_kb = max(0, total_kb - avail_kb)

    return {
        "total_kb": total_kb,
        "avail_kb": avail_kb,
        "used_kb": used_kb,
        "free_kb": free_kb,
        "cached_kb": cached_kb,
        "buffers_kb": buffers_kb,
    }


def _format_system_mem_line(metrics_kb: Dict[str, int], prefix: str) -> str:
    total = _kb_to_mib(metrics_kb["total_kb"])
    used = _kb_to_mib(metrics_kb["used_kb"])
    avail = _kb_to_mib(metrics_kb["avail_kb"])
    free = _kb_to_mib(metrics_kb["free_kb"])
    cached = _kb_to_mib(metrics_kb["cached_kb"])
    buffers = _kb_to_mib(metrics_kb["buffers_kb"])

    return (
        f"{prefix}"
        f"used={used:.1f} MiB | "
        f"avail={avail:.1f} MiB | "
        f"free={free:.1f} MiB | "
        f"cached={cached:.1f} MiB | "
        f"buffers={buffers:.1f} MiB | "
        f"total={total:.1f} MiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple ExecuTorch inferences on the Pi.")
    parser.add_argument("--pte_path", type=str, default="binary_mlp_xnnpack.pte", help="Path to .pte file.")
    parser.add_argument("--input_dim", type=int, default=4, help="Number of input bits (default: 4).")
    parser.add_argument("--num_runs", type=int, default=1000, help="Number of random inferences to run.")
    parser.add_argument(
        "--mem_sample_every",
        type=int,
        default=100,
        help="Sample system RAM every N runs (default: 100). Use 0 to disable.",
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)

    pte_path = Path(args.pte_path).expanduser().resolve()
    if not pte_path.exists():
        raise FileNotFoundError(f"Could not find .pte file at: {pte_path}")
    if args.num_runs < 1:
        raise ValueError("--num_runs must be >= 1")
    if args.mem_sample_every < 0:
        raise ValueError("--mem_sample_every must be >= 0")

    start_mem = _compute_system_mem_metrics(_read_meminfo_kb())
    print(_format_system_mem_line(start_mem, prefix="System RAM (start):     "))

    print(f"Loading ExecuTorch program: {pte_path}")

    load_start = time.perf_counter()
    rt = Runtime.get()
    program = rt.load_program(pte_path, verification=Verification.Minimal)
    forward = program.load_method("forward")
    load_ms = (time.perf_counter() - load_start) * 1000.0

    after_load_mem = _compute_system_mem_metrics(_read_meminfo_kb())
    print(_format_system_mem_line(after_load_mem, prefix="System RAM (after load):"))
    print(f"Load time:               {load_ms:.3f} ms")

    times_ms: List[float] = []
    num_correct = 0
    example = None

    # System memory samples collected during the inference loop.
    # Each entry: (run_index, used_kb, avail_kb)
    mem_samples: List[Tuple[int, int, int]] = []

    sample_every = args.mem_sample_every
    if sample_every > 0:
        metrics = _compute_system_mem_metrics(_read_meminfo_kb())
        mem_samples.append((0, metrics["used_kb"], metrics["avail_kb"]))

    for i in range(args.num_runs):
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

        if sample_every > 0:
            # Sample every N runs, including the last run.
            is_sample_point = ((i + 1) % sample_every) == 0
            is_last = (i + 1) == args.num_runs
            if is_sample_point or is_last:
                metrics = _compute_system_mem_metrics(_read_meminfo_kb())
                mem_samples.append((i + 1, metrics["used_kb"], metrics["avail_kb"]))

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
        f"runs={len(times_ms)} | "
        f"total={total_ms:.3f} ms | "
        f"avg={avg_ms:.3f} ms | "
        f"p50={p50_ms:.3f} ms | "
        f"p95={p95_ms:.3f} ms | "
        f"min={min_ms:.3f} ms | "
        f"max={max_ms:.3f} ms"
    )
    print(f"Accuracy:        {accuracy:.3%} ({num_correct}/{len(times_ms)})")

    end_mem = _compute_system_mem_metrics(_read_meminfo_kb())
    print(_format_system_mem_line(end_mem, prefix="System RAM (end):       "))

    if mem_samples:
        used_kb_vals = [x[1] for x in mem_samples]
        avail_kb_vals = [x[2] for x in mem_samples]

        used_min = min(used_kb_vals)
        used_max = max(used_kb_vals)
        used_avg = sum(used_kb_vals) / len(used_kb_vals)

        avail_min = min(avail_kb_vals)
        avail_max = max(avail_kb_vals)
        avail_avg = sum(avail_kb_vals) / len(avail_kb_vals)

        print(
            "System RAM stats: "
            f"samples={len(mem_samples)} (every {sample_every} runs) | "
            f"used_min={_kb_to_mib(used_min):.1f} MiB | "
            f"used_avg={_kb_to_mib(int(used_avg)):.1f} MiB | "
            f"used_max={_kb_to_mib(used_max):.1f} MiB | "
            f"avail_min={_kb_to_mib(avail_min):.1f} MiB | "
            f"avail_avg={_kb_to_mib(int(avail_avg)):.1f} MiB | "
            f"avail_max={_kb_to_mib(avail_max):.1f} MiB"
        )

        first_i, first_used, first_avail = mem_samples[0]
        last_i, last_used, last_avail = mem_samples[-1]
        print(
            "System RAM delta: "
            f"used_start={_kb_to_mib(first_used):.1f} MiB -> used_end={_kb_to_mib(last_used):.1f} MiB | "
            f"avail_start={_kb_to_mib(first_avail):.1f} MiB -> avail_end={_kb_to_mib(last_avail):.1f} MiB"
        )

    return


if __name__ == "__main__":
    main()
