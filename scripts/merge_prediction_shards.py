from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and validate prediction shards")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.inputs]
    merged = pd.concat(frames, ignore_index=True)
    if merged.example_id.duplicated().any():
        duplicates = merged.loc[merged.example_id.duplicated(), "example_id"].tolist()[:10]
        raise ValueError(f"Duplicate example ids across shards: {duplicates}")
    expected = set(pd.read_csv(args.manifest).example_id.astype(str))
    actual = set(merged.example_id.astype(str))
    if expected != actual:
        raise ValueError(
            f"Shard coverage mismatch: missing={len(expected - actual)}, extra={len(actual - expected)}"
        )
    merged = merged.sort_values("example_id").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Merged {len(merged)} unique predictions into {args.output}")


if __name__ == "__main__":
    main()
