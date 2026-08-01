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

