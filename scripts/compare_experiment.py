#!/usr/bin/env python3
"""Compare matched hold-out predictions with paired uncertainty estimates."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def read_predictions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()

    plain = read_predictions(args.models_dir / "plain" / "test_predictions.jsonl")
    macronized = read_predictions(
        args.models_dir / "macronized" / "test_predictions.jsonl"
    )
    if [row["id"] for row in plain] != [row["id"] for row in macronized]:
        raise ValueError("The two prediction files do not contain the same hold-out rows")

    plain_correct = np.array(
        [row["predicted_label"] == row["gold_label"] for row in plain]
    )
    macronized_correct = np.array(
        [row["predicted_label"] == row["gold_label"] for row in macronized]
    )
    paired_difference = macronized_correct.astype(float) - plain_correct.astype(float)

    rng = np.random.default_rng(args.seed)
    bootstrap = np.empty(args.bootstrap_samples)
    for index in range(args.bootstrap_samples):
        sample = rng.integers(0, len(paired_difference), len(paired_difference))
        bootstrap[index] = paired_difference[sample].mean()

    plain_only = int(np.sum(plain_correct & ~macronized_correct))
    macronized_only = int(np.sum(~plain_correct & macronized_correct))
    discordant = plain_only + macronized_only
    mcnemar_p = (
        binomtest(
            min(plain_only, macronized_only),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
        if discordant
        else 1.0
    )
    result = {
        "test_records": len(plain),
        "plain_accuracy": float(plain_correct.mean()),
        "macronized_accuracy": float(macronized_correct.mean()),
        "accuracy_difference_macronized_minus_plain": float(
            paired_difference.mean()
        ),
        "paired_bootstrap_95_percent_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "plain_only_correct": plain_only,
        "macronized_only_correct": macronized_only,
        "mcnemar_exact_p_value": float(mcnemar_p),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    output_path = args.models_dir / "comparison.json"
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
