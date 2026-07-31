#!/usr/bin/env python3
"""Train matched DistilRoBERTa metre classifiers on plain/macronized text."""

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import transformers.utils.import_utils as import_utils

# Work around an incompatible optional torchao installation in some environments.
import_utils._torchao_available = False

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


def metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    top_k = min(5, logits.shape[-1])
    top_predictions = np.argpartition(logits, -top_k, axis=-1)[:, -top_k:]
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "top5_accuracy": np.mean(
            np.any(top_predictions == labels[:, None], axis=1)
        ).item(),
    }


def train_variant(args, variant: str, labels: dict) -> None:
    set_seed(args.seed)
    variant_dir = args.data_dir / variant
    dataset = load_dataset(
        "json",
        data_files={
            split: str(variant_dir / f"{split}.jsonl")
            for split in ("train", "validation", "test")
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = dataset.map(tokenize, batched=True, desc=f"Tokenizing {variant}")
    if args.max_train_samples:
        tokenized["train"] = tokenized["train"].select(
            range(min(args.max_train_samples, len(tokenized["train"])))
        )
    if args.max_eval_samples:
        for split in ("validation", "test"):
            tokenized[split] = tokenized[split].select(
                range(min(args.max_eval_samples, len(tokenized[split])))
            )
    id_to_label = labels["id_to_label"]
    label_to_id = labels["label_to_id"]
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(id_to_label),
        id2label={index: label for index, label in enumerate(id_to_label)},
        label2id=label_to_id,
    )
    encoder_layers = model.roberta.encoder.layer
    if not 0 <= args.trainable_encoder_layers <= len(encoder_layers):
        raise ValueError(
            f"--trainable-encoder-layers must be between 0 and {len(encoder_layers)}"
        )
    for parameter in model.roberta.parameters():
        parameter.requires_grad = False
    if args.trainable_encoder_layers:
        for layer in encoder_layers[-args.trainable_encoder_layers :]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"{variant}: training {trainable:,}/{total:,} parameters "
        f"({trainable / total:.1%})"
    )

    output_dir = args.output_dir / variant
    resume_checkpoint = (
        get_last_checkpoint(str(output_dir))
        if args.resume and output_dir.exists()
        else None
    )
    if resume_checkpoint and args.trust_local_checkpoint:
        # Transformers blocks torch.load on torch<2.6 because arbitrary .pt files
        # may be unsafe. This opt-in is only for optimizer states created locally
        # by this script in the same workspace.
        import transformers.trainer as trainer_module

        trainer_module.check_torch_load_is_safe = lambda: None
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=resume_checkpoint is None,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        max_steps=args.max_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as output:
        json.dump(
            {
                "variant": variant,
                "model": args.model,
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": (
                    args.batch_size * args.gradient_accumulation_steps
                ),
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "max_length": args.max_length,
                "trainable_encoder_layers": args.trainable_encoder_layers,
                "trainable_parameters": trainable,
                "total_parameters": total,
            },
            output,
            indent=2,
        )
        output.write("\n")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(output_dir / "best_model")
    tokenizer.save_pretrained(output_dir / "best_model")
    with (output_dir / "training_history.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(
            {
                "best_metric": trainer.state.best_metric,
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "global_step": trainer.state.global_step,
                "epochs_completed": trainer.state.epoch,
                "log_history": trainer.state.log_history,
            },
            output,
            indent=2,
        )
        output.write("\n")

    test_output = trainer.predict(tokenized["test"], metric_key_prefix="test")
    test_metrics = {
        key: float(value) if isinstance(value, (float, np.floating)) else value
        for key, value in test_output.metrics.items()
    }
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as output:
        json.dump(test_metrics, output, indent=2)
        output.write("\n")

    predicted_ids = np.argmax(test_output.predictions, axis=-1)
    with (output_dir / "test_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as output:
        for record, predicted_id in zip(dataset["test"], predicted_ids):
            output.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "source": record["source"],
                        "gold_label": record["label"],
                        "predicted_label": int(predicted_id),
                        "gold_meter": record["meter"],
                        "predicted_meter": id_to_label[int(predicted_id)],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"{variant} test metrics: {json.dumps(test_metrics, indent=2)}")
    if args.cleanup_checkpoints:
        for checkpoint in output_dir.glob("checkpoint-*"):
            if checkpoint.is_dir():
                shutil.rmtree(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("plain", "macronized", "both"),
        default="both",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--model", default="distilbert/distilroberta-base"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--trainable-encoder-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trust-local-checkpoint", action="store_true")
    parser.add_argument("--cleanup-checkpoints", action="store_true")
    args = parser.parse_args()

    with (args.data_dir / "labels.json").open(encoding="utf-8") as input_file:
        labels = json.load(input_file)

    variants = ("plain", "macronized") if args.variant == "both" else (args.variant,)
    for variant in variants:
        train_variant(args, variant, labels)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
