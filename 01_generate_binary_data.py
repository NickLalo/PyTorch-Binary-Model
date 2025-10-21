#!/usr/bin/env python3
"""
Generate a tiny binary-to-integer dataset for a toy PyTorch Lightning classifier.

Creates 25 copies of each 4-bit integer (0..15), shuffles, and splits into
train and validation CSVs with columns:
    input_1, input_2, input_3, input_4, target
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


SEED = 112


def int_to_bits(n, width=4):
    """
    Convert integer n into a list of bits with fixed width (MSB first).
    """
    s = format(int(n), f"0{width}b")
    bits = [int(ch) for ch in s]
    return bits


def build_dataframe(copies_per_int=25, width=4):
    """
    Build a DataFrame with repeated binary inputs and integer targets.
    width=4 yields values 0..15.
    """
    rows = []
    max_val = (1 << width) - 1
    for value in range(max_val + 1):
        bits = int_to_bits(value, width=width)
        for _ in range(copies_per_int):
            row = {f"input_{i+1}": bits[i] for i in range(width)}
            row["target"] = value
            rows.append(row)

    df = pd.DataFrame(rows, columns=[f"input_{i+1}" for i in range(width)] + ["target"])
    return df


def shuffle_split_save(df, out_dir, train_frac=0.8, seed=SEED):
    """
    Shuffle the DataFrame, split into train/val, and save to CSV.
    """
    rng = np.random.default_rng(seed=seed)
    # pandas sample with random_state for deterministic order
    df_shuf = df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)

    n_total = len(df_shuf)
    n_train = int(n_total * train_frac)
    df_train = df_shuf.iloc[:n_train].reset_index(drop=True)
    df_val = df_shuf.iloc[n_train:].reset_index(drop=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)

    print(f"Wrote {len(df_train)} rows to {train_path}")
    print(f"Wrote {len(df_val)} rows to {val_path}")
    return


def main():
    parser = argparse.ArgumentParser(description="Generate binary-to-integer CSVs for a toy classifier.")
    parser.add_argument("--copies", type=int, default=25, help="Copies per integer value.")
    parser.add_argument("--width", type=int, default=4, help="Number of input bits. 4 covers values 0..15.")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Training split fraction.")
    parser.add_argument("--out_dir", type=str, default="training_data", help="Output directory for CSV files.")
    args = parser.parse_args()

    np.random.seed(SEED)

    df = build_dataframe(copies_per_int=args.copies, width=args.width)
    print(f"Built dataset with shape {df.shape} (width={args.width}, copies_per_int={args.copies})")
    print(df.head())

    shuffle_split_save(df, args.out_dir, train_frac=args.train_frac, seed=SEED)
    return


if __name__ == "__main__":
    main()
