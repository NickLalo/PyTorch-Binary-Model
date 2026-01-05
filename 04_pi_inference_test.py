#!/usr/bin/env python3
"""
Pi inference smoke test for ExecuTorch .pte model.

What it does:
  1) Loads binary_mlp_xnnpack.pte from the current directory (or --pte_path).
  2) Creates one random binary input vector of length --input_dim.
  3) Runs ExecuTorch forward() once.
  4) Prints the input bits, logits, predicted class, and the correct integer.

Notes:
  - This script assumes your exported model expects a float32 tensor shaped (1, input_dim).
  - Default input_dim is 4 (for 4-bit integers 0..15).
"""

import argparse
from pathlib import Path
from typing import List

import torch
from executorch.runtime import Runtime, Verification


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
    parser = argparse.ArgumentParser(description="Run a single ExecuTorch inference on the Pi.")
    parser.add_argument("--pte_path", type=str, default="binary_mlp_xnnpack.pte", help="Path to .pte file.")
    parser.add_argument("--input_dim", type=int, default=4, help="Number of input bits (default: 4).")
    args = parser.parse_args()

    torch.manual_seed(SEED)

    pte_path = Path(args.pte_path).expanduser().resolve()
    if not pte_path.exists():
        raise FileNotFoundError(f"Could not find .pte file at: {pte_path}")

    print(f"Loading ExecuTorch program: {pte_path}")

    rt = Runtime.get()
    program = rt.load_program(pte_path, verification=Verification.Minimal)
    forward = program.load_method("forward")

    bits_i64 = torch.randint(0, 2, (args.input_dim,), dtype=torch.int64)
    bits_list = bits_i64.tolist()
    expected = bits_to_int(bits_list)

    x = bits_i64.to(dtype=torch.float32).unsqueeze(0).contiguous()

    print(f"Input bits:      {format_bits(bits_list)}")
    print(f"Correct integer: {expected}")

    out = forward.execute((x,))[0]

    if not isinstance(out, torch.Tensor):
        out = torch.tensor(out)

    logits = out.to(dtype=torch.float32).cpu()

    pred = int(torch.argmax(logits, dim=1).item())

    print(f"Logits:          {logits.numpy().tolist()}")
    print(f"Predicted class: {pred}")
    print(f"Correct:         {pred == expected}")

    return


if __name__ == "__main__":
    main()
