#!/usr/bin/env python3
"""Remove under-supported labels from two aligned JSONL datasets."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain", type=Path, default=Path("train.jsonl"))
    parser.add_argument(
        "--macronized", type=Path, default=Path("train_macronized.jsonl")
    )
    parser.add_argument("--minimum-count", type=int, default=10)
    parser.add_argument(
        "--meter-regex",
        help="Optional full-match regular expression that retained labels must satisfy",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plain = read_jsonl(args.plain)
    macronized = read_jsonl(args.macronized)
    if len(plain) != len(macronized):
        raise ValueError("Input datasets have different lengths")

    counts = Counter(record["meter"] for record in plain)
    meter_pattern = re.compile(args.meter_regex) if args.meter_regex else None
    retained_labels = {
        meter
        for meter, count in counts.items()
        if count >= args.minimum_count
        and (meter_pattern is None or meter_pattern.fullmatch(meter))
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "plain": args.output_dir / "train.jsonl",
        "macronized": args.output_dir / "train_macronized.jsonl",
    }
    with paths["plain"].open(
        "w", encoding="utf-8"
    ) as plain_output, paths["macronized"].open(
        "w", encoding="utf-8"
    ) as macronized_output:
        retained = 0
        for index, (plain_record, macronized_record) in enumerate(
            zip(plain, macronized), 1
        ):
            if plain_record["meter"] != macronized_record["meter"]:
                raise ValueError(f"Meter mismatch at row {index}")
            if plain_record["meter"] not in retained_labels:
                continue
            plain_output.write(json.dumps(plain_record, ensure_ascii=False) + "\n")
            macronized_output.write(
                json.dumps(macronized_record, ensure_ascii=False) + "\n"
            )
            retained += 1

    print(
        f"Retained {retained}/{len(plain)} records and "
        f"{len(retained_labels)}/{len(counts)} labels "
        f"(minimum count: {args.minimum_count}, regex: {args.meter_regex!r})"
    )


if __name__ == "__main__":
    main()
