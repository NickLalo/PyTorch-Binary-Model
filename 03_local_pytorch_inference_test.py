#!/usr/bin/env python3
"""
Load the trained binary classifier checkpoint and time per-sample inference on random binary inputs.
"""

import argparse
import importlib.util
import time
from pathlib import Path
from typing import List, Optional

import torch


def load_training_module() -> object:
    """Dynamically load the training script so we can reuse BinaryClassifier."""
    script_path = Path(__file__).resolve().parent / "02_train_binary_classifier.py"
    spec = importlib.util.spec_from_file_location("binary_training_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_checkpoint(explicit_path: Optional[str] = None) -> Path:
    """
    Resolve the checkpoint to load. Preference:
    1) --ckpt_path if provided
    2) best_model_*.ckpt in cwd
    """
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    # if no explicit path, search cwd for best_model_*.ckpt
    cwd = Path.cwd()
    candidates = sorted(cwd.glob("best_model_*.ckpt"))
    if not candidates:
        raise FileNotFoundError("No checkpoint found. Provide --ckpt_path or place a best_model_*.ckpt in the project root.")
    return candidates[0].resolve()


def bits_to_int(bits: List[int]) -> int:
    """MSB-first binary list to integer."""
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def format_bits(bits: List[int]) -> str:
    return " ".join(str(int(b)) for b in bits)


def run_inference_tests(model: torch.nn.Module, num_runs: int) -> None:
    start_time = time.time()
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    input_dim = int(model.hparams.input_dim)
    correct_count = 0
    durations_ms = []
    for i in range(num_runs):
        bits = torch.randint(0, 2, (input_dim,), dtype=torch.int64)
        expected = bits_to_int(bits.tolist())
        x = bits.float().unsqueeze(0).to(device)

        with torch.no_grad():
            start = time.perf_counter()
            logits = model(x)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            pred = int(torch.argmax(logits, dim=1).item())

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


def main():
    print(f"Running inference tests on binary classifier...")
    parser = argparse.ArgumentParser(description="Run random inference timing tests.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to Lightning checkpoint.")
    parser.add_argument("--runs", type=int, default=10_000, help="Number of random inputs to test.")
    args = parser.parse_args()

    training_module = load_training_module()
    ckpt_path = find_checkpoint(args.ckpt_path)
    print(f"Loading checkpoint: {ckpt_path}")
    model = training_module.BinaryClassifier.load_from_checkpoint(ckpt_path)

    run_inference_tests(model, num_runs=args.runs)


if __name__ == "__main__":
    main()
