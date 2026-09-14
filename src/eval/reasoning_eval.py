"""Exact-match evaluation on the synthetic reasoning test set.

Used three ways, all through the same code path so the numbers are comparable:

  * **pretrained, zero-shot** -- the Phase 2 checkpoint given a bare question.
    Expected to score near nothing; it has never seen the answer format. The
    point of reporting it is to show that, not to flatter the finetune.
  * **pretrained, few-shot** -- the same checkpoint with ``k`` worked examples
    prepended. This is the honest baseline: it supplies the output format, so
    whatever gap remains is about reasoning rather than about formatting.
  * **finetuned** -- zero-shot again, since the format is now learned.

Batched decoding and absolute positions. Both models use *learned absolute*
positional embeddings, so a token's representation depends on its index from the
left. Left-padding a batch to a common width -- the usual trick for batched
generation -- would therefore shift every prompt's positions and change the
model's input, silently. Prompts are instead bucketed by exact token length and
batched within a bucket, so nothing is padded and every position index is the
one the model would see decoding that prompt alone. Prompt lengths here span
roughly 24-67 tokens, so buckets stay large enough to be worth batching.

Answer extraction reads the span after the final answer prefix, which is the
same fixed marker the generator emits (``उत्तर:``). Under the rationale style
the model must produce its chain first and the marker last, so taking the final
occurrence scores the model's conclusion rather than anything it restated on
the way there.

Usage:
    python -m src.eval.reasoning_eval \
        --checkpoint hindi/checkpoints/finetune_best.pt \
        --model-config hindi/configs/model.yaml \
        --tokenizer tokenizers/hindi/hindi_bpe_8k.model \
        --test data/hindi/reasoning/test.jsonl \
        --language hindi --mode finetuned --answer-style direct \
        --out hindi/eval/reasoning_finetuned.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from src.eval import provenance
from src.model.transformer_lm import TransformerLM, TransformerLMConfig
from src.reasoning.answers import extract_answer, normalise

# Generation budget per answer style. "direct" needs only the answer line;
# "rationale" must also emit the restated chain before it.
MAX_NEW = {"direct": 12, "rationale": 48}

# Answer parsing lives in src/reasoning/answers.py, next to the generator that
# defines the format; re-exported here because this is where callers look.
__all__ = ["extract_answer", "normalise", "greedy_continuations", "score",
           "MAX_NEW"]


def build_fewshot_preamble(demos: list, language: str,
                           answer_style: str) -> str:
    """Concatenate ``k`` worked examples into an in-context preamble.

    Args:
        demos: Records from the *train* split, so the test split's held-out
            entity pool is never shown to the model.
        language: Unused beyond documenting intent; the records carry their own
            rendered text.
        answer_style: Which answer field the demonstrations should show.

    Returns:
        A string ending in a blank line, to be prefixed to each test prompt.
    """
    key = "answer_direct" if answer_style == "direct" else "answer_rationale"
    return "".join(f"{d['prompt']}{d[key]}\n\n" for d in demos)


@torch.no_grad()
def greedy_continuations(model, sp, prompts: list, max_new: int, device: str,
                         eos_id: int, batch_size: int, ctx_len: int) -> list:
    """Greedy-decode a continuation for every prompt.

    Prompts are grouped by exact tokenised length so batches need no padding
    (see the module docstring on absolute positions). Results are returned in
    the caller's original order.

    Args:
        model: A ``TransformerLM`` in eval mode.
        sp: SentencePiece processor.
        prompts: Prompt strings.
        max_new: Tokens to generate per prompt.
        device: Device string.
        eos_id: Stops a batch once every member has emitted this.
        batch_size: Prompts per forward pass, within a length bucket.
        ctx_len: Model context; over-long prompts are left-truncated so the
            question survives even if early few-shot demonstrations do not.

    Returns:
        One decoded continuation string per input prompt, prompt text excluded.
    """
    model.eval()
    encoded = [sp.encode(p) for p in prompts]
    encoded = [ids[-(ctx_len - max_new):] if len(ids) > ctx_len - max_new else ids
               for ids in encoded]

    buckets = defaultdict(list)
    for i, ids in enumerate(encoded):
        buckets[len(ids)].append(i)

    out = [None] * len(prompts)
    for length, idxs in buckets.items():
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            x = torch.tensor([encoded[i] for i in chunk], dtype=torch.long,
                             device=device)
            gen = model.generate(x, max_new_tokens=max_new, greedy=True,
                                 eos_id=eos_id)
            for row, i in enumerate(chunk):
                new_ids = gen[row, length:].tolist()
                if eos_id in new_ids:
                    new_ids = new_ids[:new_ids.index(eos_id)]
                out[i] = sp.decode(new_ids)
    return out


def score(records: list, predictions: list, language: str) -> dict:
    """Exact match overall and broken down by every axis the report needs.

    Args:
        records: Test records, carrying ``answer`` and ``meta``.
        predictions: Raw continuations, aligned with ``records``.
        language: For answer-prefix extraction.

    Returns:
        A dict with ``exact_match``, ``format_rate`` (how often the answer
        prefix was produced at all), and per-family / per-hop / per-attribute /
        seen-vs-held-out-phrasing breakdowns.
    """
    hits = defaultdict(int)
    total = defaultdict(int)
    formatted = 0
    per_example = []

    for rec, cont in zip(records, predictions):
        pred = extract_answer(cont, language)
        gold = normalise(rec["answer"])
        ok = bool(pred) and pred == gold
        formatted += bool(pred)
        m = rec["meta"]
        for k in (("all",), ("family", rec["family"]), ("hops", m["hops"]),
                  ("attribute", m["attribute"]),
                  ("phrasing", "held_out" if m["held_out_phrasing"] else "seen")):
            key = k[0] if len(k) == 1 else f"{k[0]}::{k[1]}"
            total[key] += 1
            hits[key] += ok
        per_example.append({"family": rec["family"], "gold": gold,
                            "pred": pred, "correct": ok,
                            "hops": m["hops"],
                            "continuation": cont[:200]})

    def rate(prefix):
        """Accuracy per value of one breakdown axis."""
        return {k.split("::", 1)[1]: round(hits[k] / total[k], 4)
                for k in sorted(total) if k.startswith(prefix + "::")}

    n = total["all"]
    return {
        "n": n,
        "exact_match": round(hits["all"] / n, 4),
        "format_rate": round(formatted / n, 4),
        "by_family": rate("family"),
        "by_hops": {str(k): v for k, v in sorted(rate("hops").items())},
        "by_attribute": rate("attribute"),
        "by_phrasing": rate("phrasing"),
        "confusions": Counter(
            f"{e['gold']} -> {e['pred'] or '<no answer>'}"
            for e in per_example if not e["correct"]).most_common(15),
        "_per_example": per_example,
    }


def main():
    """Parse the CLI, decode the test set, and write the scored result."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--train", default=None,
                    help="train split, required for --mode fewshot demos")
    ap.add_argument("--language", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--mode", required=True,
                    choices=["zeroshot", "fewshot", "finetuned"])
    ap.add_argument("--answer-style", default="direct",
                    choices=["direct", "rationale"])
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N test examples")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-examples", type=int, default=40,
                    help="qualitative successes/failures to keep in the JSON")
    args = ap.parse_args()

    import random
    import sentencepiece as spm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

    cfg = TransformerLMConfig.from_yaml(args.model_config)
    model = TransformerLM(cfg).to(device)
    from src.train.checkpoint import load_checkpoint
    meta = load_checkpoint(args.checkpoint, model, map_location="cpu",
                           restore_rng=False)
    print(f"loaded {args.checkpoint} (step {meta['step']:,}) on {device}")

    records = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    if args.limit:
        records = records[:args.limit]

    preamble = ""
    if args.mode == "fewshot":
        if not args.train:
            ap.error("--mode fewshot requires --train for demonstrations")
        pool = [json.loads(l) for l in open(args.train, encoding="utf-8")]
        rng = random.Random(args.seed)
        preamble = build_fewshot_preamble(rng.sample(pool, args.shots),
                                          args.language, args.answer_style)
        print(f"few-shot preamble: {args.shots} demos, "
              f"{len(sp.encode(preamble))} tokens")

    prompts = [preamble + r["prompt"] for r in records]
    # eos lives in the model YAML's tokenizer block, not in the architecture
    # dataclass; fall back to what the SentencePiece model itself declares.
    import yaml
    tok_block = yaml.safe_load(open(args.model_config, encoding="utf-8")) \
        .get("tokenizer", {})
    eos_id = tok_block.get("eos_id", sp.eos_id())

    conts = greedy_continuations(model, sp, prompts,
                                 MAX_NEW[args.answer_style], device, eos_id,
                                 args.batch_size, cfg.ctx_len)

    result = score(records, conts, args.language)
    per_ex = result.pop("_per_example")
    k = args.dump_examples // 2
    result["examples"] = {
        "correct": [e for e in per_ex if e["correct"]][:k],
        "incorrect": [e for e in per_ex if not e["correct"]][:k],
    }
    result["setup"] = {
        "mode": args.mode, "answer_style": args.answer_style,
        "shots": args.shots if args.mode == "fewshot" else 0,
        "language": args.language, "checkpoint_step": meta["step"],
        "max_new_tokens": MAX_NEW[args.answer_style],
        # Which split was scored. Needed downstream: "test" and "test_seen"
        # differ only in their entity pool, and a comparison table that could
        # not tell them apart would present the control as the real result.
        "test_split": Path(args.test).stem,
        "seed": args.seed,
    }
    result["provenance"] = provenance.build(
        "src.eval.reasoning_eval",
        inputs={"checkpoint": args.checkpoint, "test": args.test,
                "tokenizer": args.tokenizer})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(f"\n  mode {args.mode:<10} exact match {result['exact_match']:.4f}  "
          f"format rate {result['format_rate']:.4f}  (n={result['n']:,})")
    print("  by hops: " + "  ".join(
        f"{h}:{v:.3f}" for h, v in result["by_hops"].items()))
    print(f"  -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
