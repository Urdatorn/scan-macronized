# scan-macronized

An experiment measuring how predicted Ancient Greek vowel-length annotation
affects downstream hexameter classification, for the paper "Automatic annotation of Ancient Greek vowel length".

## Dataset

- 52,119 aligned verses from 74 represented source files
- 31 metre classes
- Every class occurs at least 10 times
- Metres match `^(--|-uu|uuuu){5}--$`
- Split sizes: 41,695 train / 5,212 validation / 5,212 test
- Plain and macronized variants reuse identical record IDs and split assignments

The macronized text was produced with
`oga-macronizer-char`.

## 30-epoch result

Both classifiers use `distilbert/distilroberta-base`, with the final encoder
layer and classification head trainable (7.7M of 82.1M parameters). Best
checkpoints were selected by validation accuracy.

Test accuracy on 5,212 verses. Gains are given as absolute percentage points
(pp) and, for reference, as relative improvement over the plain baseline.

| Seed | Plain | Macronized | Absolute gain | Relative gain |
|---|---|---|---|---|
| 13 | 71.9% | 78.4% | 6.45 pp | 8.96% |
| 42 | 72.5% | 77.8% | 5.22 pp | 7.19% |
| 2026 | 71.2% | 76.9% | 5.70 pp | 8.00% |
| **Mean ± 95% t-interval** | | | **5.79 ± 1.54 pp** | 8.05 ± 2.20% |

The figure reported in the paper (5.8% ± 1.5%) is the absolute gain in
percentage points. Per-seed values are in
`models_30epochs/seed_*/comparison.json`.

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

