"""Add a random 'likes' column to a CSV file.

Example:
  python src/youtube_csv_process.py --input data/raw/english_youtube.csv --output data/raw/english_youtube_processed.csv --col likes --min 0 --max 500 --seed 42 --overwrite (to overwrite input file)
"""
from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
import os


def add_random_likes(input_path: str, output_path: str | None = None, col: str = "likes", low: int = 0, high: int = 1000, seed: int | None = None, overwrite: bool = False):
    df = pd.read_csv(input_path)
    rng = np.random.default_rng(seed)
    df[col] = rng.integers(low, high + 1, size=len(df))

    if output_path is None:
        if overwrite:
            output_path = input_path
        else:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_with_{col}{ext}"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True, help="Input CSV file path")
    p.add_argument("-o", "--output", help="Output CSV file path (default: adds suffix)")
    p.add_argument("--col", default="likes", help="Name of the likes column to add")
    p.add_argument("--min", type=int, default=0, help="Minimum likes (inclusive)")
    p.add_argument("--max", type=int, default=1000, help="Maximum likes (inclusive)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (optional)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite input file if set")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = add_random_likes(args.input, args.output, col=args.col, low=args.min, high=args.max, seed=args.seed, overwrite=args.overwrite)
    print(f"Wrote: {out}")