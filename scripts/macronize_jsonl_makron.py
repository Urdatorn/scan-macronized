#!/usr/bin/env python3
"""Macronize the text column of a JSONL dataset with Makron (Urdatorn/Makron-seed*).

Uses the batched MacronPredictor from a checkout of Urdatorn/Makron-code.

    python3 scripts/macronize_jsonl_makron.py data/plain/train.jsonl \
        data_makron/macronized/train.jsonl --code /path/to/Makron-code
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--code", type=Path, required=True, help="Checkout of Urdatorn/Makron-code")
    parser.add_argument("--model", default="Urdatorn/Makron-seed1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    sys.path.insert(0, str(args.code))
    from predict import MacronPredictor

    predictor = MacronPredictor(args.model, device=args.device)

    with args.input.open(encoding="utf-8") as input_file:
        records = [json.loads(line) for line in input_file]
    for line_number, record in enumerate(records, 1):
        if not isinstance(record.get("text"), str):
            raise ValueError(f"{args.input}:{line_number}: text is not a string")

    texts = predictor.macronize_lines([record["text"] for record in records], batch_size=args.batch_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for record, text in zip(records, texts):
            record["text"] = text
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {args.output} ({args.model})")


if __name__ == "__main__":
    main()
