"""Text generation from Model H and Model L under several decoding settings.

Brief section 2.3 asks for samples from held-out prefixes under greedy decoding
and temperature sampling at 0.5 / 1.0 / 1.5, and section 2.4 then scores those
samples against the true continuations. Both jobs are served by one pass here:
for every prompt this module records the prefix, the *reference* continuation
that actually followed it in the held-out corpus, and one generation per
decoding setting. ``src.eval.metrics`` consumes the JSON without re-running the
model, so the expensive GPU step happens exactly once.

Prompt selection is a single-pass reservoir sample over the test split, seeded,
so the same prompts come back on every run and neither model is quietly given an
easier set than the other. Prompts are truncated to exactly ``--prompt-tokens``
tokens, which makes the batch rectangular and means every sample starts from the
same amount of context.

Decoding is batched: one forward pass extends all prompts together. Sampling
draws come from a seeded generator on the model's device, so a re-run reproduces
the sampled text and not merely its statistics.

Usage:
    python -m src.eval.generate \
        --checkpoint <drive>/hindi/checkpoints/checkpoint_best.pt \
        --model-config hindi/configs/model.yaml \
        --tokenizer tokenizers/hindi/hindi_bpe_8k.model \
        --test-jsonl <drive>/hindi/splits/test.jsonl \
        --language hindi \
        --out hindi/samples/generations.json \
        --markdown hindi/samples/generations.md
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sentencepiece as spm

from src.eval import provenance
from src.model.transformer_lm import TransformerLM, TransformerLMConfig
from src.train.checkpoint import load_checkpoint


def reservoir_sample_docs(jsonl_path, n: int, min_chars: int, seed: int) -> list:
    """Draw ``n`` documents uniformly at random from a JSONL split in one pass.

    The test splits run to hundreds of megabytes, so the file is streamed rather
    than loaded. Reservoir sampling gives a uniform draw over every document that
    clears ``min_chars`` while holding only ``n`` of them in memory, and seeding
    the generator makes the draw reproducible -- both models are prompted from
    the same positions of their own held-out data.

    Args:
        jsonl_path: ``test.jsonl`` from Phase 1, one document per line.
        n: Reservoir size.
        min_chars: Skip documents shorter than this, so a prompt plus its
            reference continuation actually fits.
        seed: Seed for the sampling generator.

    Returns:
        (docs, n_eligible) -- the drawn ``{"doc_id", "source", "text"}`` dicts,
        and how many documents cleared ``min_chars`` in total.
    """
    rng = random.Random(seed)
    reservoir, seen = [], 0

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            if len(text) < min_chars:
                continue
            seen += 1
            if len(reservoir) < n:
                reservoir.append(rec)
            else:
                # Classic algorithm R: the k-th eligible item replaces a random
                # slot with probability n/k, which leaves every item equally likely.
                j = rng.randrange(seen)
                if j < n:
                    reservoir[j] = rec

    return [{"doc_id": r.get("doc_id"), "source": r.get("source"),
             "text": r.get("text", "")} for r in reservoir], seen


def build_prompts(docs: list, sp, prompt_tokens: int, ref_tokens: int,
                  n_prompts: int) -> list:
    """Encode candidate documents into fixed-length prompts plus references.

    Args:
        docs: Candidate documents from :func:`reservoir_sample_docs`.
        sp: A loaded SentencePiece processor.
        prompt_tokens: Tokens of prefix fed to the model.
        ref_tokens: Tokens of true continuation kept as the reference.
        n_prompts: How many prompts to keep.

    Returns:
        Up to ``n_prompts`` dicts with the prompt ids, prompt text, reference
        ids and reference text. Documents that fall short after encoding are
        skipped -- the character filter is only an approximation of token length.
    """
    out = []
    need = prompt_tokens + ref_tokens
    for d in docs:
        ids = sp.encode(d["text"], out_type=int)
        if len(ids) < need:
            continue
        p_ids, r_ids = ids[:prompt_tokens], ids[prompt_tokens:need]
        out.append({
            "doc_id": d["doc_id"], "source": d["source"],
            "prompt_ids": p_ids, "prompt_text": sp.decode(p_ids),
            "reference_ids": r_ids, "reference_text": sp.decode(r_ids),
        })
        if len(out) == n_prompts:
            break
    return out


@torch.no_grad()
def generate_batch(model, prompt_ids: torch.Tensor, max_new_tokens: int,
                   greedy: bool, temperature: float, eos_id: int,
                   generator: torch.Generator) -> torch.Tensor:
    """Extend a rectangular batch of prompts by ``max_new_tokens`` tokens.

    This mirrors :meth:`TransformerLM.generate` but threads an explicit
    generator through ``torch.multinomial`` so that a sampled run is
    reproducible, and it never stops early: sequences that emit ``<eos>`` keep
    being extended and are trimmed afterwards, which keeps the batch rectangular
    without letting one finished sequence cut another one short.

    Args:
        model: A ``TransformerLM`` in eval mode.
        prompt_ids: (B, P) prompt token ids.
        max_new_tokens: Tokens to append.
        greedy: Take the argmax instead of sampling.
        temperature: Softmax temperature; ignored when ``greedy``.
        eos_id: End-of-document id, recorded for later trimming.
        generator: Seeded generator for the sampling draws.

    Returns:
        (B, P + max_new_tokens) token ids, prompt included.
    """
    model.eval()
    idx = prompt_ids
    ctx = model.config.ctx_len

    for _ in range(max_new_tokens):
        logits, _, _ = model(idx[:, -ctx:])
        logits = logits[:, -1, :].float()             # (B, V), last position
        if greedy:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator)
        idx = torch.cat([idx, nxt], dim=1)

    return idx


def trim_at_eos(ids: list, eos_id: int) -> tuple[list, bool]:
    """Cut a continuation at its first ``<eos>``.

    A document boundary is where the model said the text ends; scoring past it
    would measure the start of an unrelated document. Returns the trimmed ids
    and whether an ``<eos>`` was actually produced -- the fraction of samples
    that terminate on their own is itself worth reporting.
    """
    if eos_id in ids:
        return ids[:ids.index(eos_id)], True
    return ids, False


def decoding_settings(temperatures: list) -> list:
    """The decoding configurations required by the brief, greedy first."""
    settings = [{"name": "greedy", "greedy": True, "temperature": 0.0}]
    for t in temperatures:
        settings.append({"name": f"temp_{t:g}", "greedy": False, "temperature": t})
    return settings


def write_markdown(path, language: str, records: list, settings: list,
                   max_show: int) -> None:
    """Write a human-readable sample sheet for the report appendix."""
    lines = [f"# Generated samples -- {language}", "",
             f"{len(records)} held-out prompts, {len(settings)} decoding settings. "
             f"The first {max_show} are shown; the full set is in the JSON "
             f"alongside this file.", ""]
    for rec in records[:max_show]:
        lines += [f"## Prompt {rec['index']} (`{rec['source']}`)", "",
                  "**Prompt**", "", f"> {rec['prompt_text']}", "",
                  "**Reference continuation**", "",
                  f"> {rec['reference_text']}", ""]
        for s in settings:
            g = rec["generations"][s["name"]]
            lines += [f"**{s['name']}** ({g['n_tokens']} tokens"
                      f"{', hit <eos>' if g['hit_eos'] else ''})", "",
                      f"> {g['text']}", ""]
        lines.append("---")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    """Sample held-out prompts and generate continuations under each setting.

    Draws prompts by seeded reservoir sampling over the test split, generates
    under greedy decoding and each requested temperature, and writes prompts,
    true continuations and generations to one JSON so the metrics step needs
    no second GPU pass. Optionally also writes a readable Markdown sheet.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--language", required=True)
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperatures", default="0.5,1.0,1.5")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--markdown-samples", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TransformerLMConfig.from_yaml(args.model_config)
    model = TransformerLM(cfg).to(device)
    meta = load_checkpoint(args.checkpoint, model, map_location="cpu",
                           restore_rng=False)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    eos_id = sp.eos_id()

    print(f"{args.language}: checkpoint step {meta['step']:,}, device {device}, "
          f"eos_id {eos_id}")

    # Oversample candidates 4x: the character filter only approximates token
    # length, so some draws are discarded once encoded.
    need = args.prompt_tokens + args.max_new_tokens
    min_chars = int(need * 2.5)
    docs, n_eligible = reservoir_sample_docs(
        args.test_jsonl, args.n_prompts * 4, min_chars, args.seed)
    prompts = build_prompts(docs, sp, args.prompt_tokens,
                            args.max_new_tokens, args.n_prompts)
    print(f"  sampled {len(prompts)} prompts from {n_eligible:,} eligible "
          f"held-out documents")
    if len(prompts) < args.n_prompts:
        print(f"  warning: wanted {args.n_prompts}, kept {len(prompts)}")

    settings = decoding_settings([float(t) for t in args.temperatures.split(",")])
    records = [{"index": i, "doc_id": p["doc_id"], "source": p["source"],
                "prompt_text": p["prompt_text"], "prompt_ids": p["prompt_ids"],
                "reference_text": p["reference_text"],
                "reference_ids": p["reference_ids"], "generations": {}}
               for i, p in enumerate(prompts)]

    t0 = time.time()
    for s in settings:
        # One generator per setting, reseeded from the run seed, so adding or
        # removing a temperature does not change the others' samples.
        gen = torch.Generator(device=device).manual_seed(args.seed)
        for start in range(0, len(prompts), args.batch_size):
            chunk = prompts[start:start + args.batch_size]
            x = torch.tensor([c["prompt_ids"] for c in chunk],
                             dtype=torch.long, device=device)
            full = generate_batch(model, x, args.max_new_tokens, s["greedy"],
                                  s["temperature"], eos_id, gen)
            cont = full[:, x.size(1):].tolist()
            for k, ids in enumerate(cont):
                trimmed, hit = trim_at_eos(ids, eos_id)
                records[start + k]["generations"][s["name"]] = {
                    "ids": trimmed, "text": sp.decode(trimmed),
                    "n_tokens": len(trimmed), "hit_eos": hit,
                    "temperature": s["temperature"], "greedy": s["greedy"]}
        done = [r["generations"][s["name"]] for r in records]
        print(f"  {s['name']:10} mean {sum(g['n_tokens'] for g in done)/len(done):6.1f} "
              f"tokens, {sum(g['hit_eos'] for g in done)}/{len(done)} hit <eos>")

    payload = {
        "provenance": provenance.build(
            "src.eval.generate",
            {"checkpoint": args.checkpoint, "model_config": args.model_config,
             "tokenizer": args.tokenizer, "test_jsonl": args.test_jsonl},
            {"torch": torch.__version__, "device": device}),
        "language": args.language, "checkpoint": str(args.checkpoint),
        "checkpoint_step": meta["step"], "tokenizer": str(args.tokenizer),
        "seed": args.seed, "n_prompts": len(records),
        "prompt_tokens": args.prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "settings": [s["name"] for s in settings],
        "eos_id": eos_id, "elapsed_sec": round(time.time() - t0, 1),
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"wrote {out} ({payload['elapsed_sec']}s)")

    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.markdown, args.language, records, settings,
                       args.markdown_samples)
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
