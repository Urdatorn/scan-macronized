#!/usr/bin/env python3
"""Run the counterbalanced three-seed metre-classification experiment."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("models_10epochs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2026])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    script = Path(__file__).with_name("train_meter_classifier.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule = []
    for position, seed in enumerate(args.seeds):
        # Counterbalance model order to reduce systematic thermal/order effects.
        variants = (
            ("plain", "macronized")
            if position % 2 == 0
            else ("macronized", "plain")
        )
        for variant in variants:
            schedule.append({"seed": seed, "variant": variant})

    with (args.output_dir / "schedule.json").open("w", encoding="utf-8") as output:
        json.dump(schedule, output, indent=2)
        output.write("\n")

    for run in schedule:
        seed = run["seed"]
        variant = run["variant"]
        seed_dir = args.output_dir / f"seed_{seed}"
        metrics_path = seed_dir / variant / "test_metrics.json"
        predictions_path = seed_dir / variant / "test_predictions.jsonl"
        if metrics_path.exists() and predictions_path.exists():
            print(f"Skipping completed run: seed={seed}, variant={variant}")
            continue
        command = [
            sys.executable,
            str(script),
            "--variant",
            variant,
            "--output-dir",
            str(seed_dir),
            "--seed",
            str(seed),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            "16",
            "--gradient-accumulation-steps",
            "1",
            "--resume",
            "--trust-local-checkpoint",
            "--cleanup-checkpoints",
            "--data-dir",
            str(args.data_dir),
        ]
        print(f"Starting seed={seed}, variant={variant}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
