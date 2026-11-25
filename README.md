# scan-macronized
[![DOI](https://img.shields.io/badge/DOI-10.57967%2Fhf%2F11854-blue)](https://doi.org/10.57967/hf/11854)

An experiment measuring how predicted Ancient Greek vowel-length annotation
affects downstream hexameter classification, for the paper "Automatic annotation of Ancient Greek vowel length".

## Dataset

- 52,119 aligned verses from 74 represented source files
- 31 metre classes
- Every class occurs at least 10 times
- Metres match `^(--|-uu|uuuu){5}--$`
- Split sizes: 41,695 train / 5,212 validation / 5,212 test
- Plain and macronized variants reuse identical record IDs and split assignments

There are two macronized versions of the data:

- `data_makron/macronized/` (current, used in the paper): macronized with
  [Urdatorn/Makron-seed1](https://huggingface.co/Urdatorn/Makron-seed1)
  by `scripts/macronize_jsonl_makron.py`. `data_makron/plain` links to `data/plain`.
- `data/macronized/` (earlier): macronized with `Ericu950/oga-macronizer-char`
  by `scripts/macronize_jsonl.py`.

The two macronizations differ in 20% of the lines, mostly by more long marks
in the Makron version (train: 21,853 long marks against 20,321).

## 30-epoch result (Makron-seed1, used in the paper)

Both classifiers use `distilbert/distilroberta-base`, with the final encoder
layer and classification head trainable (7.7M of 82.1M parameters). Best
checkpoints were selected by validation accuracy. Both variants were trained
on one Intel Arc GPU (XPU) with the counterbalanced schedule of
`scripts/run_multiseed_experiment.py`:

```bash
python3 scripts/run_multiseed_experiment.py --output-dir models_30epochs_makron \
  --epochs 30 --data-dir data_makron
python3 scripts/aggregate_multiseed.py --models-dir models_30epochs_makron --epochs 30
```

Test accuracy on 5,212 verses. Gains are given as absolute percentage points
(pp) and, for reference, as relative improvement over the plain baseline.

| Seed | Plain | Macronized | Absolute gain | Relative gain | McNemar p |
|---|---|---|---|---|---|
| 13 | 72.4% | 80.7% | 8.25 pp | 11.39% | 4e-48 |
| 42 | 72.2% | 78.1% | 5.93 pp | 8.21% | 7e-24 |
| 2026 | 72.3% | 78.5% | 6.22 pp | 8.60% | 7e-27 |
| **Mean ± 95% t-interval** | 72.3% | 79.1% | **6.80 ± 3.14 pp** | 9.40 ± 4.31% |

The hierarchical bootstrap 95% interval of the gain is 5.3–8.3 pp. Best
checkpoints fell between epochs 22 and 29. Per-seed values are in
`models_30epochs_makron/seed_*/comparison.json`, the aggregate in
`models_30epochs_makron/aggregate_summary.json`. The classifier weights
of both experiments (`best_model/model.safetensors`, 318 MB each) are on the
Hugging Face Hub at [Urdatorn/Makron-scan](https://huggingface.co/Urdatorn/Makron-scan),
which mirrors this repository; the Makron-rerun weights are not on GitHub,
since its Git LFS budget is used up.

## Training

```bash
python3 scripts/train_meter_classifier.py \
  --variant both \
  --output-dir models \
  --seed 13 \
  --epochs 30 \
  --batch-size 16 \
  --gradient-accumulation-steps 1 \
  --cleanup-checkpoints
```

