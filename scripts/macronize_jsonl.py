#!/usr/bin/env python3
"""Macronize the text column of a JSONL dataset with oga-macronizer-char."""

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Checkout of Urdatorn/grc-macronizer",
    )
    parser.add_argument("--model", default="Ericu950/oga-macronizer-char")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    sys.path[:0] = [str(args.source / "macron_model"), str(args.source)]

    # Some environments contain an incompatible optional torchao installation.
    import transformers.utils.import_utils as import_utils

    import_utils._torchao_available = False

    from predict import MacronPredictor
    from tokenizer import MACRON_LONG, MACRON_SHORT, decode, encode_plain

    predictor = MacronPredictor(args.model)
    device = predictor.device
    model = predictor.model
    vocab = predictor.vocab
    max_len = predictor.max_len

    def macronize_batch(texts: list[str]) -> list[str]:
        encoded = [encode_plain(text, vocab, fit=False) for text in texts]
        longest = max((len(item["plane1"]) for item in encoded), default=0)
        if longest > max_len:
            return [predictor.macronize_line(text) for text in texts]

        plane1 = torch.zeros((len(texts), longest), dtype=torch.long, device=device)
        plane2 = torch.zeros((len(texts), longest), dtype=torch.long, device=device)
        attention = torch.zeros((len(texts), longest), dtype=torch.long, device=device)
        for index, item in enumerate(encoded):
            length = len(item["plane1"])
            plane1[index, :length] = torch.tensor(item["plane1"], device=device)
            plane2[index, :length] = torch.tensor(item["plane2"], device=device)
            attention[index, :length] = 1

        with torch.no_grad():
            predictions = model(
                plane1_ids=plane1,
                plane2_ids=plane2,
                attention_mask=attention,
            ).logits.argmax(-1).cpu().tolist()

        results = []
        for item, labels in zip(encoded, predictions):
            labels = labels[: len(item["chars"])]
            safe_labels = [
                label if eligible and label in (MACRON_SHORT, MACRON_LONG) else 0
                for label, eligible in zip(labels, item["dichrona_mask"])
            ]
            results.append(decode(item["chars"], safe_labels))
        return results

    records: list[dict[str, object]] = []
    with args.input.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            record = json.loads(line)
            if not isinstance(record.get("text"), str):
                raise ValueError(f"{args.input}:{line_number}: text is not a string")
            records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            macronized = macronize_batch([record["text"] for record in batch])
            for record, text in zip(batch, macronized):
                record["text"] = text
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            done = min(start + args.batch_size, len(records))
            print(f"\rMacronized {done}/{len(records)} records", end="", flush=True)
    print(f"\nWrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
