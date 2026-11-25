#!/usr/bin/env python3
"""Aggregate matched plain/macronized results across random seeds."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "sample_sd": float(np.std(values, ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=Path("models_10epochs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2026])
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    metric_names = (
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_top5_accuracy",
    )
    per_seed = []
    correctness_differences = []
    reference_ids = None

    for seed in args.seeds:
        seed_dir = args.models_dir / f"seed_{seed}"
        metrics = {
            variant: read_json(seed_dir / variant / "test_metrics.json")
            for variant in ("plain", "macronized")
        }
        predictions = {
            variant: read_jsonl(
                seed_dir / variant / "test_predictions.jsonl"
            )
            for variant in ("plain", "macronized")
        }
        ids = [row["id"] for row in predictions["plain"]]
        if ids != [row["id"] for row in predictions["macronized"]]:
            raise ValueError(f"Prediction IDs differ within seed {seed}")
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f"Prediction IDs differ across seeds at seed {seed}")

        plain_correct = np.array(
            [
                row["predicted_label"] == row["gold_label"]
                for row in predictions["plain"]
            ]
        )
        macronized_correct = np.array(
            [
                row["predicted_label"] == row["gold_label"]
                for row in predictions["macronized"]
            ]
        )
        plain_only = int(np.sum(plain_correct & ~macronized_correct))
        macronized_only = int(np.sum(~plain_correct & macronized_correct))
        discordant = plain_only + macronized_only
        mcnemar_p = (
            binomtest(
                min(plain_only, macronized_only),
                n=discordant,
                p=0.5,
            ).pvalue
            if discordant
            else 1.0
        )
        difference = macronized_correct.astype(float) - plain_correct.astype(float)
        correctness_differences.append(difference)
        per_seed.append(
            {
                "seed": seed,
                "plain": {name: metrics["plain"][name] for name in metric_names},
                "macronized": {
                    name: metrics["macronized"][name] for name in metric_names
                },
                "accuracy_gain": float(difference.mean()),
                "plain_only_correct": plain_only,
                "macronized_only_correct": macronized_only,
                "mcnemar_exact_p_value": float(mcnemar_p),
            }
        )

    aggregate = {}
    for metric_name in metric_names:
        aggregate[metric_name] = {
            variant: mean_sd(
                [run[variant][metric_name] for run in per_seed]
            )
            for variant in ("plain", "macronized")
        }

    gains = np.array([run["accuracy_gain"] for run in per_seed])
    gain_mean = float(gains.mean())
    gain_sd = float(gains.std(ddof=1))
    t_margin = float(t.ppf(0.975, df=len(gains) - 1) * gain_sd / np.sqrt(len(gains)))

    differences = np.stack(correctness_differences)
    rng = np.random.default_rng(args.bootstrap_seed)
    hierarchical_bootstrap = np.empty(args.bootstrap_samples)
    for index in range(args.bootstrap_samples):
        seed_sample = rng.integers(0, differences.shape[0], differences.shape[0])
        example_sample = rng.integers(0, differences.shape[1], differences.shape[1])
        hierarchical_bootstrap[index] = differences[seed_sample][
            :, example_sample
        ].mean()

    result = {
        "seeds": args.seeds,
        "test_records_per_seed": len(reference_ids),
        "epochs": args.epochs,
        "per_seed": per_seed,
        "aggregate_metrics": aggregate,
        "accuracy_gain_macronized_minus_plain": {
            "mean": gain_mean,
            "sample_sd_across_seeds": gain_sd,
            "seed_level_t_95_percent_ci": [
                gain_mean - t_margin,
                gain_mean + t_margin,
            ],
            "hierarchical_bootstrap_95_percent_ci": [
                float(np.quantile(hierarchical_bootstrap, 0.025)),
                float(np.quantile(hierarchical_bootstrap, 0.975)),
            ],
            "hierarchical_bootstrap_samples": args.bootstrap_samples,
        },
    }
    output_path = args.models_dir / "aggregate_summary.json"
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
