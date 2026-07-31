#!/usr/bin/env python3
"""Create aligned, source-stratified splits for the macronization experiment."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def source_sequence(source_dir: Path, records: list[dict]) -> list[str]:
    """Map a possibly label-filtered record subsequence back to source filenames."""
    sources = []
    record_index = 0
    for path in sorted(source_dir.glob("*.txt")):
        with path.open(encoding="utf-8-sig") as input_file:
            for line in input_file:
                line = line.strip()
                if not line:
                    continue
                text, meter = line[1:-1].rsplit("], [", 1)
                source_record = {
                    "text": text,
                    "meter": "".join(meter.split()),
                }
                if (
                    record_index < len(records)
                    and records[record_index] == source_record
                ):
                    sources.append(path.name)
                    record_index += 1
    if record_index != len(records):
        raise ValueError(
            f"Could map only {record_index}/{len(records)} records to source files"
        )
    return sources


def stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def assign_splits(sources: list[str], meters: list[str], seed: int) -> list[str]:
    by_source: dict[str, list[int]] = defaultdict(list)
    for index, source in enumerate(sources):
        by_source[source].append(index)

    # Reserve one deterministic example of every category for training. Without
    # this, singleton/rare categories can land only in validation or test and are
    # impossible for a supervised classifier to learn.
    by_meter: dict[str, list[int]] = defaultdict(list)
    for index, meter in enumerate(meters):
        by_meter[meter].append(index)
    required_train = {
        min(indices, key=lambda index: stable_seed(seed, f"label:{index}"))
        for indices in by_meter.values()
    }

    assignments = [""] * len(sources)
    for source, indices in sorted(by_source.items()):
        random.Random(stable_seed(seed, source)).shuffle(indices)
        size = len(indices)
        validation_size = round(size * 0.1)
        test_size = round(size * 0.1)
        train_size = size - validation_size - test_size
        reserved = [index for index in indices if index in required_train]
        unreserved = [index for index in indices if index not in required_train]
        if len(reserved) > train_size:
            raise ValueError(
                f"{source} needs {len(reserved)} label reservations but has only "
                f"{train_size} training slots"
            )
        train_indices = reserved + unreserved[: train_size - len(reserved)]
        remainder = unreserved[train_size - len(reserved) :]
        for index in train_indices:
            assignments[index] = "train"
        for index in remainder[:validation_size]:
            assignments[index] = "validation"
        for index in remainder[validation_size:]:
            assignments[index] = "test"
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain", type=Path, default=Path("train.jsonl"))
    parser.add_argument(
        "--macronized", type=Path, default=Path("train_macronized.jsonl")
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    plain = read_jsonl(args.plain)
    macronized = read_jsonl(args.macronized)
    sources = source_sequence(args.source_dir, plain)
    if not len(plain) == len(macronized) == len(sources):
        raise ValueError(
            f"Record counts differ: plain={len(plain)}, "
            f"macronized={len(macronized)}, sources={len(sources)}"
        )

    for index, (plain_record, macronized_record) in enumerate(
        zip(plain, macronized), 1
    ):
        if plain_record["meter"] != macronized_record["meter"]:
            raise ValueError(f"Meter mismatch at record {index}")
        if any(character.isspace() for character in plain_record["meter"]):
            raise ValueError(f"Whitespace remains in meter at record {index}")

    labels = sorted({record["meter"] for record in plain})
    label_to_id = {label: index for index, label in enumerate(labels)}
    assignments = assign_splits(
        sources, [record["meter"] for record in plain], args.seed
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant in ("plain", "macronized"):
        (args.output_dir / variant).mkdir(exist_ok=True)

    handles = {
        (variant, split): (args.output_dir / variant / f"{split}.jsonl").open(
            "w", encoding="utf-8"
        )
        for variant in ("plain", "macronized")
        for split in ("train", "validation", "test")
    }
    assignment_path = args.output_dir / "split_assignments.jsonl"
    with assignment_path.open("w", encoding="utf-8") as assignment_file:
        try:
            for record_id, (plain_record, macronized_record, source, split) in enumerate(
                zip(plain, macronized, sources, assignments)
            ):
                label = label_to_id[plain_record["meter"]]
                common = {
                    "id": record_id,
                    "meter": plain_record["meter"],
                    "label": label,
                    "source": source,
                }
                for variant, source_record in (
                    ("plain", plain_record),
                    ("macronized", macronized_record),
                ):
                    output_record = {"text": source_record["text"], **common}
                    handles[(variant, split)].write(
                        json.dumps(output_record, ensure_ascii=False) + "\n"
                    )
                assignment_file.write(
                    json.dumps(
                        {"id": record_id, "source": source, "split": split}
                    )
                    + "\n"
                )
        finally:
            for handle in handles.values():
                handle.close()

    split_counts = Counter(assignments)
    label_counts = Counter(record["meter"] for record in plain)
    split_labels = {
        split: {
            plain[index]["meter"]
            for index, assigned_split in enumerate(assignments)
            if assigned_split == split
        }
        for split in ("train", "validation", "test")
    }
    metadata = {
        "seed": args.seed,
        "split_strategy": (
            "Within each source file, deterministic shuffled 80/10/10; "
            "one example per label is reserved for training"
        ),
        "num_records": len(plain),
        "num_sources": len(set(sources)),
        "num_labels": len(labels),
        "split_counts": dict(split_counts),
        "split_label_counts": {
            split: len(values) for split, values in split_labels.items()
        },
        "validation_labels_absent_from_train": len(
            split_labels["validation"] - split_labels["train"]
        ),
        "test_labels_absent_from_train": len(
            split_labels["test"] - split_labels["train"]
        ),
        "singleton_labels": sum(count == 1 for count in label_counts.values()),
        "labels_with_fewer_than_10_records": sum(
            count < 10 for count in label_counts.values()
        ),
    }
    with (args.output_dir / "labels.json").open("w", encoding="utf-8") as output:
        json.dump(
            {
                "label_to_id": label_to_id,
                "id_to_label": labels,
                "counts": {label: label_counts[label] for label in labels},
            },
            output,
            ensure_ascii=False,
            indent=2,
        )
        output.write("\n")
    with (args.output_dir / "dataset_info.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(metadata, output, indent=2)
        output.write("\n")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
