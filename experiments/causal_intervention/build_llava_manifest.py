from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CONDITIONS = ("canonical", "content_only", "position_only", "joint")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_csv(args.input)
    source = source[
        source.boundary_status.eq("cross_tile") & source.intervention.eq("canonical")
    ].copy()
    if source.empty:
        raise ValueError("No cross-tile canonical rows found")
    expected_rows = len(source) * len(CONDITIONS)
    frames = []
    for condition in CONDITIONS:
        part = source.copy()
        part["position_intervention"] = condition
        part["merged_token_permutation"] = "2,0,3,1"
        part["tile_permutation"] = "0,1,2,3"
        part["intervention"] = "llava_token_" + condition
        part["example_id"] = part["example_id"].str.replace(
            "_s4_canonical", "_s4_lcausal_" + condition, regex=False
        )
        frames.append(part)
    result = pd.concat(frames, ignore_index=True).sort_values("example_id")
    if len(result) != expected_rows or result.example_id.duplicated().any():
        raise ValueError("LLaVA causal manifest invariant failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
