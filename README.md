# scan-macronized

An experiment measuring how predicted Ancient Greek vowel-length annotation
affects downstream hexameter classification.

The repository contains aligned plain and macronized datasets, deterministic
source-stratified train/validation/test splits, preprocessing and training
scripts, and the best models from the seed-13 30-epoch experiment.

## Dataset

- 52,119 aligned verses from 74 represented source files
- 31 metre classes
- Every class occurs at least 10 times
- Metres match `^(--|-uu|uuuu){5}--$`
- Split sizes: 41,695 train / 5,212 validation / 5,212 test
- Plain and macronized variants reuse identical record IDs and split assignments

The macronized text was produced with
[`Ericu950/oga-macronizer-char`](https://huggingface.co/Ericu950/oga-macronizer-char).

## Seed-13 30-epoch result

Both classifiers use `distilbert/distilroberta-base`, with the final encoder
layer and classification head trainable (7.7M of 82.1M parameters). Best
checkpoints were selected by validation accuracy.

| Held-out test metric | Plain | Macronized |
|---|---:|---:|
| Accuracy | 71.91% | 78.36% |
| Top-5 accuracy | 97.47% | 98.60% |
| Macro-F1 | 0.5348 | 0.6267 |
| Weighted F1 | 0.7067 | 0.7736 |

Macronization improved accuracy by 6.45 percentage points. The paired bootstrap
95% confidence interval is +5.33 to +7.58 points; exact McNemar
`p = 1.98e-28`.

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

On macOS, prefix the command with `caffeinate -dimsu env
PYTORCH_ENABLE_MPS_FALLBACK=1` to prevent idle sleep and permit CPU fallback
for unsupported MPS operations.

The final `model.safetensors` files are stored with Git LFS. Intermediate
training checkpoints and earlier exploratory model runs are intentionally
excluded.
