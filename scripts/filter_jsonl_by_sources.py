#!/usr/bin/env python3
"""Filter aligned JSONL datasets to the files present in a selected source directory."""

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def source_sequence(source_dir: Path) -> list[str]:
    sequence = []
    for path in sorted(source_dir.glob("*.txt")):
        with path.open(encoding="utf-8-sig") as input_file:
            sequence.extend(path.name for line in input_file if line.strip())
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain", type=Path, default=Path("train.jsonl"))
    parser.add_argument(
        "--macronized", type=Path, default=Path("train_macronized.jsonl")
    )
    parser.add_argument(
        "--complete-source-dir",
        type=Path,
        required=True,
        help="Source directory corresponding to every row in the input JSONLs",
    )
    parser.add_argument(
        "--selected-source-dir",
        type=Path,
        required=True,
        help="Only basenames present here are retained",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plain = read_jsonl(args.plain)
    macronized = read_jsonl(args.macronized)
    sources = source_sequence(args.complete_source_dir)
    if not len(plain) == len(macronized) == len(sources):
        raise ValueError(
            f"Counts differ: plain={len(plain)}, macronized={len(macronized)}, "
            f"source rows={len(sources)}"
        )

    selected = {path.name for path in args.selected_source_dir.glob("*.txt")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "plain": args.output_dir / "train.jsonl",
        "macronized": args.output_dir / "train_macronized.jsonl",
    }
    with output_paths["plain"].open(
        "w", encoding="utf-8"
    ) as plain_output, output_paths["macronized"].open(
        "w", encoding="utf-8"
    ) as macronized_output:
        count = 0
        for plain_record, macronized_record, source in zip(
            plain, macronized, sources
        ):
            if plain_record["meter"] != macronized_record["meter"]:
                raise ValueError(f"Meter mismatch in source {source}")
            if source not in selected:
                continue
            plain_output.write(
                json.dumps(plain_record, ensure_ascii=False) + "\n"
            )
            macronized_output.write(
                json.dumps(macronized_record, ensure_ascii=False) + "\n"
            )
            count += 1
    print(
        f"Wrote {count} aligned records from {len(selected)} selected source files "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
