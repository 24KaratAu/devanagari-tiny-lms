# Model L — `nepali`

Everything specific to Model L lives here. The Transformer implementation itself is
shared with Model H under `src/` — the two models are architecturally
identical (both Phase 1 vocabularies came out at exactly 8,000), so duplicating the
code would mean maintaining two copies of the same 25,379,840-parameter network.

**The models share no data, tokenizer, vocabulary, or weights.** Only the
implementation is common; every artifact below is this language's alone.

## Contents

| Path | What |
| :--- | :--- |
| `configs/model.yaml` | Architecture — 25,379,840 params |
| `configs/train.yaml` | Optimizer, schedule, data and checkpoint paths |
| `logs/train.jsonl` | Per-step training log; the loss curve is plotted from this |
| `figures/attention_layer{0,4,7}.png` | Attention heatmaps, heads 0/1/4/7, on a real held-out passage |
| `figures/attention_head_summary.png` | Per-head mean distance and normalised entropy, all 8 layers |
| `figures/attention_passage.txt` | The passage the heatmaps are drawn on, token by token |
| `eval/intrinsic.json` | Cross-entropy, perplexity, BPB, bits-per-char — full val and test |
| `eval/attention_stats.json` | Per-head entropy, distance, previous-token and sink mass (64 heads) |
| `samples/generations.json` | 32 held-out prompts × 4 decoding settings, with the true continuations |
| `samples/generations.md` | The same, formatted for reading |

Loss curves and the generation-metric tables cover both models at once and live
in [`report/figures/`](../report/figures); the analysis is in
[`report/phase2_report.md`](../report/phase2_report.md).

Tokenizer: `tokenizers/nepali/nepali_bpe_8k.model` (Phase 1 deliverable, kept at its
Phase 1 path so the Phase 1 report's references stay valid).

## Corpus

Token counts **measured** on 2026-09-02 by tokenizing every byte of the split
files with the Phase 1 model — not estimated.

| Split | Tokens |
| :--- | ---: |
| train | 569,684,621 |
| val | 31,676,467 |
| test | 31,385,814 |
| **total** | **632,746,902** |

Measured chars/token: **4.126**.

## Training

Platform: **Kaggle (T4)**. One pass over the training split = **17,429 steps**
at 64 x 512 = 32,768 tokens per step.

```bash
python -m src.train.trainer --config nepali/configs/train.yaml
```

Resume is automatic: the trainer looks for `checkpoint_latest.pt` in the checkpoint
directory and continues from it, restoring model, optimizer, scheduler, GradScaler,
step, and RNG state. Nothing extra to pass.

## Checkpoints

Drive: `gdrive:24KaratAu/nepali/checkpoints/`
(shared as *Anyone with the link -> Viewer*)

| File | Cadence |
| :--- | :--- |
| `checkpoint_latest.pt` | every 500 steps |
| `checkpoint_best.pt` | on validation-loss improvement |
| `checkpoint_step_N.pt` | every 2,000 steps |
| `config.json` | hyperparameters, saved alongside the weights |

Each is ~305 MB (101.5 MB fp32 weights + AdamW moment buffers) and contains model
weights, optimizer state, scheduler state, training step, and configuration, per
the brief's mandatory checkpointing requirement.

> **Drive links to be filled in once the run completes.**
