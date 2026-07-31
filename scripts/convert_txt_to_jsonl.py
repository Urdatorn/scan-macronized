#!/usr/bin/env python3
"""Combine bracket-delimited text files into a two-column JSONL dataset."""

import argparse
import json
from pathlib import Path


def parse_record(line: str, source: Path, line_number: int) -> dict[str, str]:
    line = line.strip()
    if not line:
        raise ValueError("blank line")
    if not line.startswith("[") or not line.endswith("]"):
        raise ValueError(f"{source}:{line_number}: missing outer brackets")

    boundary = "], ["
    if boundary not in line:
        raise ValueError(f"{source}:{line_number}: missing {boundary!r} separator")

    text, meter = line[1:-1].rsplit(boundary, 1)
    return {"text": text, "meter": "".join(meter.split())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    sources = sorted(args.input_dir.glob("*.txt"))
    if not sources:
        raise SystemExit(f"No .txt files found in {args.input_dir}")

    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for source in sources:
            with source.open(encoding="utf-8-sig") as input_file:
                for line_number, line in enumerate(input_file, 1):
                    if not line.strip():
                        continue
                    record = parse_record(line, source, line_number)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1

    print(f"Wrote {count} records from {len(sources)} files to {args.output}")


if __name__ == "__main__":
    main()
