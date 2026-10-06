# Phase 3 model comparison

- Protocol: docs/EVAL_METHODOLOGY.md § 8
- Slice: 500 images, 732 references (about 1.46 per image), fingerprint `6b5628bfa410ed233ef9603beed3c05c9e63acebc8bfa31d7e63e634f9e25116`
- Normalisation: preprocess_caption -> strip_sentinels

> The slice comes from COCO train2017. The CNN + Transformer held these images out of training; the Hugging Face baselines were fine-tuned on COCO training data (ViT-GPT2 possibly), so they may have seen them. These scores are not a held-out, like-for-like comparison (docs/EVAL_METHODOLOGY.md § 8.5).

| Run | Model | Decoding | Revision | Samples | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | METEOR | CIDEr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `phase3-blip-base-greedy` | blip-base | greedy | `82a3776` | 500 | 56.61 | 39.70 | 27.86 | 19.88 | 42.13 | 17.23 | 1.06 |
| `phase3-git-base-coco-greedy` | git-base-coco | greedy | `a13141d` | 500 | 51.59 | 36.55 | 26.05 | 18.83 | 47.08 | 21.92 | 1.46 |
| `stabilized-beam-w4-lp07-rp12` (reference) | inceptionv3-transformer-stabilized | beam | — | 500 | 41.93 | 25.41 | 16.01 | 10.39 | 36.84 | 15.56 | 0.83 |
| `stabilized-greedy` (reference) | inceptionv3-transformer-stabilized | greedy | — | 500 | 42.20 | 26.09 | 16.52 | 10.57 | 37.57 | 15.45 | 0.79 |
| `phase3-vit-gpt2-greedy` | vit-gpt2 | greedy | `dc68f91` | 500 | 49.12 | 33.41 | 22.91 | 15.84 | 44.51 | 19.78 | 1.26 |

Values are rounded to two decimals; `comparison.json` holds the exact values from each run's `metrics.json`.
