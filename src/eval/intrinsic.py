"""Intrinsic language-modelling metrics: cross-entropy, perplexity, bits-per-byte.

Required by brief section 2.3:
  * validation cross-entropy loss and perplexity, PPL = exp(CE);
  * bits-per-byte on the same held-out text, "to make the two languages more
    comparable when tokenizers differ".

    PPL = exp(CE_nats)
    BPB = CE_nats / (ln 2 x bytes_per_token)

BPB divides out the tokenizer by re-expressing the loss per byte of underlying
UTF-8 text rather than per token. It does not divide out the *script*: Devanagari
costs ~3 UTF-8 bytes per character where Latin costs 1, so a Devanagari model's
BPB reads far lower than an English one's at equal quality. Bits-per-character is
reported alongside for that reason -- it is the fairer of the two here, since both
languages share the script but not the tokenizer.

Evaluation strides through the split in non-overlapping ``ctx_len`` windows, so
every token is predicted exactly once and none is scored twice. Tokens in the
first position of each window are predicted from a single token of context; with
ctx_len 512 that is 1/512 of positions, and the same convention applies to both
models so the comparison stays fair.

Usage:
    python -m src.eval.intrinsic \
        --checkpoint <drive>/hindi/checkpoints/checkpoint_best.pt \
        --model-config hindi/configs/model.yaml \
        --tokens-dir <drive>/hindi/tokens --language hindi \
        --out hindi/eval/intrinsic.json
"""

import argparse
import json
import math
from pathlib import Path

import torch

from src.eval import provenance
from src.model.transformer_lm import TransformerLM, TransformerLMConfig
from src.train.checkpoint import load_checkpoint
from src.train.dataset import iter_eval_batches

LN2 = math.log(2)


def load_sidecar(tokens_dir, language: str, split: str) -> dict:
    """Read the packing statistics for one split, refusing un-repaired byte counts.

    An early version of the packer counted whole JSONL lines -- including the
    ``doc_id`` / ``source`` / ``text_sha256`` envelope, roughly 45% of a record --
    as if they were text. That inflates bytes/token and understates BPB by about
    6%. Sidecars written before the fix are rejected here rather than silently
    producing a flattering number.

    Args:
        tokens_dir: Directory holding ``{language}_{split}.json``.
        language: "hindi" or "nepali".
        split: "val" or "test".

    Returns:
        The sidecar dict.

    Detection is by marker key, not by threshold: a broken Hindi sidecar reads
    2.69 bytes/char against a correct 2.54, and both sit under the 3.0 ceiling
    for UTF-8 Devanagari, so no numeric bound separates them. Sidecars written by
    the fixed packer carry ``n_bytes_jsonl``; ones corrected after the fact carry
    ``bytes_repaired_at``. A sidecar with neither predates the fix.

    Raises:
        ValueError: if the sidecar predates the byte-counting fix.
    """
    p = Path(tokens_dir) / f"{language}_{split}.json"
    stats = json.loads(p.read_text(encoding="utf-8"))

    if "n_bytes_jsonl" not in stats and "bytes_repaired_at" not in stats:
        raise ValueError(
            f"{p.name} predates the byte-counting fix: n_bytes_utf8 still counts "
            f"the JSON envelope (doc_id, source, text_sha256 -- ~45% of a record), "
            f"which understates BPB by about 6%. Repair it with:\n"
            f"  python -m src.data.pack_tokens --repair-bytes "
            f"--input-dir <splits> --tokenizer <model> "
            f"--output-dir {tokens_dir} --language {language} --splits {split}")

    bpc = stats["n_bytes_utf8"] / stats["n_chars"]
    if bpc > 3.0:
        raise ValueError(
            f"{p.name}: {bpc:.3f} bytes/char exceeds 3.0, the maximum for UTF-8 "
            f"Devanagari -- the byte count cannot be text alone.")
    stats["bytes_per_token_checked"] = stats["n_bytes_utf8"] / stats["n_tokens_no_eos"]
    return stats


@torch.no_grad()
def evaluate_split(model, bin_path, ctx_len: int, batch_size: int,
                   device: str, amp_dtype=None, max_tokens: int = None) -> dict:
    """Token-weighted mean cross-entropy over a split.

    Averaging must be weighted by token count, not by batch: the final batch is
    usually short, and treating it as a full one biases the result.

    Args:
        model: A ``TransformerLM``.
        bin_path: Packed ``.bin`` for the split.
        ctx_len: Window length.
        batch_size: Windows per forward pass.
        device: Device string.
        amp_dtype: Autocast dtype, or None for full precision.
        max_tokens: Score only the first N tokens of the split instead of all
            of it. None (the default) scores the whole split, which is what the
            Phase 2 headline numbers used -- leaving them reproducible
            byte-for-byte. A budget exists because Phase 3 measures this metric
            four times (two checkpoints x two languages) purely to detect
            forgetting, where a 2M-token slice separates "unchanged" from
            "destroyed" just as well as 26M does and costs a minute instead of
            twelve.

    Returns:
        {"cross_entropy", "n_tokens", "n_windows"}.
    """
    model.eval()
    total_nats, total_tokens, windows = 0.0, 0, 0

    for x, y in iter_eval_batches(bin_path, ctx_len, batch_size, max_tokens,
                                  device):
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            _, loss, _ = model(x, targets=y)
        n = y.numel()
        total_nats += loss.item() * n        # loss is a mean over y.numel()
        total_tokens += n
        windows += x.size(0)

    return {"cross_entropy": total_nats / total_tokens,
            "n_tokens": total_tokens, "n_windows": windows}


def derive_metrics(cross_entropy: float, stats: dict) -> dict:
    """Turn a cross-entropy in nats/token into the reported metrics.

    Args:
        cross_entropy: Mean CE in nats per token.
        stats: Packing sidecar, for the byte and character counts.

    Returns:
        CE in nats and bits, perplexity, BPB, bits-per-character, and the
        normalisation constants used, so every number in the report is traceable.
    """
    bpt = stats["n_bytes_utf8"] / stats["n_tokens_no_eos"]
    cpt = stats["n_chars"] / stats["n_tokens_no_eos"]
    return {
        "cross_entropy_nats": cross_entropy,
        "cross_entropy_bits": cross_entropy / LN2,
        "perplexity": math.exp(cross_entropy),
        "bits_per_byte": cross_entropy / (LN2 * bpt),
        "bits_per_char": cross_entropy / (LN2 * cpt),
        "bytes_per_token": bpt,
        "chars_per_token": cpt,
        "n_bytes_utf8": stats["n_bytes_utf8"],
        "n_chars": stats["n_chars"],
        "n_docs": stats["n_docs"],
    }


def main():
    """Score one model on its held-out splits and write the metrics JSON.

    Loads the checkpoint, evaluates each requested split end to end, derives
    perplexity, bits-per-byte and bits-per-character from the packing sidecar,
    and writes the result with a provenance block naming every input.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--tokens-dir", required=True)
    ap.add_argument("--language", required=True)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="score only the first N tokens of each split; omit "
                         "for the full split, as the Phase 2 results used")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.float16 if device == "cuda" else None

    cfg = TransformerLMConfig.from_yaml(args.model_config)
    model = TransformerLM(cfg).to(device)
    meta = load_checkpoint(args.checkpoint, model, map_location="cpu",
                           restore_rng=False)
    print(f"{args.language}: checkpoint step {meta['step']:,}, "
          f"{model.num_params():,} parameters, device {device}")

    inputs = {"checkpoint": args.checkpoint, "model_config": args.model_config}
    for split in args.splits.split(","):
        inputs[f"{split}_bin"] = Path(args.tokens_dir) / f"{args.language}_{split}.bin"
        inputs[f"{split}_sidecar"] = Path(args.tokens_dir) / f"{args.language}_{split}.json"

    results = {
        "provenance": provenance.build(
            "src.eval.intrinsic", inputs,
            {"torch": torch.__version__, "device": device,
             "amp_dtype": str(amp), "batch_size": args.batch_size}),
        "language": args.language, "checkpoint": str(args.checkpoint),
        "checkpoint_step": meta["step"], "n_params": model.num_params(),
        "ctx_len": cfg.ctx_len, "vocab_size": cfg.vocab_size,
        "max_tokens": args.max_tokens, "splits": {}}

    for split in args.splits.split(","):
        stats = load_sidecar(args.tokens_dir, args.language, split)
        bin_path = Path(args.tokens_dir) / f"{args.language}_{split}.bin"
        run = evaluate_split(model, bin_path, cfg.ctx_len, args.batch_size,
                             device, amp, max_tokens=args.max_tokens)
        m = derive_metrics(run["cross_entropy"], stats)
        m.update(n_tokens_scored=run["n_tokens"], n_windows=run["n_windows"],
                 token_budget=args.max_tokens,
                 full_split=args.max_tokens is None)
        results["splits"][split] = m

        print(f"  {split:5} CE {m['cross_entropy_nats']:.4f} nats  "
              f"PPL {m['perplexity']:7.2f}  BPB {m['bits_per_byte']:.4f}  "
              f"bits/char {m['bits_per_char']:.4f}  "
              f"({run['n_tokens']:,} tokens scored)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
