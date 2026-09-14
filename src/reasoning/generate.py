"""Generate a language's synthetic reasoning corpus: train / val / test JSONL.

Run once per language. Nothing crosses between the two runs -- separate entity
inventories, separate surface realisations, separate output directories -- so
Model H and Model L are finetuned on corpora that share no data, exactly as the
pretraining corpora did.

Leakage control, in the order it is applied:

  1. **Held-out entities.** Test examples draw names only from the disjoint test
     pool in ``entities.py``. A model that memorised "राम wins" during
     finetuning gains nothing: राम never appears in the test set. Validation
     deliberately shares the train pool, because it selects checkpoints and
     selecting on out-of-distribution entities would measure generalisation
     instead of convergence.
  2. **Held-out phrasing.** A configurable fraction of test examples uses the
     surface realisations reserved in ``templates.py`` and never emitted into
     train or val. Every example records ``held_out_phrasing`` so accuracy can
     be reported separately for seen and unseen phrasing -- two different
     generalisation demands that a single pooled number would hide.
  3. **Exact-prompt deduplication.** One hash set spans all three splits, so no
     prompt string can appear in more than one, and repeated draws inside a
     split are discarded too. The set is seeded by generating test first, so a
     collision is always resolved in favour of keeping the test item.

Determinism. Everything derives from ``--seed`` through a single
``random.Random``; re-running with the same seed reproduces the corpus
byte-for-byte. The generated data goes to Drive (per the TA's answer in
``doubts-answers.md``); this script is what makes that upload reproducible.

Usage:
    python -m src.reasoning.generate --language hindi \
        --out-dir data/hindi/reasoning --train 40000 --val 5000 --test 5000
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from src.eval import provenance
from src.reasoning import templates
from src.reasoning.entities import ATTRIBUTES, pool
from src.reasoning.templates import FAMILIES, render

# Distinct quantities must differ by more than a rounding artefact, or a
# "which is larger" question becomes a coin flip dressed up as reasoning.
MIN_GAP = {"height": 3, "age": 2, "weight": 3, "marks": 4, "price": 25,
           "length": 5}


def sample_values(attribute: str, language: str, n: int, rng,
                  ties: int = 0) -> list:
    """Draw ``n`` quantities on an attribute's scale, well separated.

    Args:
        attribute: Key of ``ATTRIBUTES[language]``.
        language: Language whose scale to use.
        n: How many quantities.
        rng: A ``random.Random``.
        ties: How many of the values should duplicate another one. Used by the
            equality family; 0 everywhere else.

    Returns:
        A list of ``n`` ints, shuffled, with all distinct values separated by at
        least ``MIN_GAP[attribute]`` and exactly ``ties`` duplicates.
    """
    lo, hi, step = ATTRIBUTES[language][attribute]["value_range"]
    gap = MIN_GAP[attribute]
    grid = list(range(lo, hi + 1, step))
    distinct = n - ties
    for _ in range(200):
        picked = sorted(rng.sample(grid, distinct))
        if all(b - a >= gap for a, b in zip(picked, picked[1:])):
            break
    else:                       # scale too tight for the gap; fall back evenly
        span = (hi - lo) // max(distinct - 1, 1)
        picked = [lo + i * span for i in range(distinct)]
    values = picked + [rng.choice(picked) for _ in range(ties)]
    rng.shuffle(values)
    return values


def make_example(family: str, language: str, split: str, rng,
                 held_out_phrasing: bool):
    """Sample one scenario and render it.

    Args:
        family: A key of ``FAMILIES``.
        language: ``"hindi"`` or ``"nepali"``.
        split: ``"train"``, ``"val"`` or ``"test"``; chooses the entity pool.
        rng: A ``random.Random``.
        held_out_phrasing: Passed through to the renderer.

    Returns:
        A ``templates.Example``.
    """
    spec = FAMILIES[family]
    attribute = rng.choice(list(ATTRIBUTES[language]))
    domain = ATTRIBUTES[language][attribute]["domain"]
    entities = rng.sample(pool(language, domain, split), spec["n"])
    # Equality has to be reachable as an *answer*, not just assertable as a
    # premise. Relation questions ask about the two rank-extremes, and with one
    # duplicate among three values those two are never the tied pair -- so a
    # single tie leaves "बराबर" unreachable. Collapsing the whole chain (ties =
    # n - 1) does make the endpoints equal, and turns the item into transitivity
    # of equality rather than of inequality.
    ties = 0
    if family == "eq3_relation":
        r = rng.random()
        ties = 2 if r < 0.25 else 1 if r < 0.60 else 0
    elif family == "val2_relation" and rng.random() < 0.18:
        ties = 1
    values = sample_values(attribute, language, spec["n"], rng, ties)
    return render(family, language, attribute, entities, values, rng,
                  held_out_phrasing)


def build_split(language: str, split: str, n: int, rng, seen: set,
                held_out_frac: float) -> list:
    """Generate ``n`` deduplicated examples for one split.

    Args:
        language: Language to generate.
        split: Split name; ``"test"`` is the only one that reaches the held-out
            entity pool and the held-out phrasings.
        n: Target example count.
        rng: A ``random.Random``.
        seen: Prompt hashes already emitted, mutated in place. Shared across
            splits so no prompt can straddle two of them.
        held_out_frac: Fraction of test examples that use reserved phrasing.

    Returns:
        A list of ``n`` ``Example`` objects with distinct prompts.

    Raises:
        RuntimeError: If the template space is too small to fill the split
            without duplicates.
    """
    families = list(FAMILIES)
    out, attempts, budget = [], 0, n * 60
    while len(out) < n:
        attempts += 1
        if attempts > budget:
            raise RuntimeError(
                f"{split}: only {len(out):,}/{n:,} distinct prompts after "
                f"{attempts:,} draws -- the template space is exhausted. "
                f"Reduce the split size or widen the entity pools.")
        fam = families[len(out) % len(families)]       # exact family balance
        held = split == "test" and rng.random() < held_out_frac
        ex = make_example(fam, language, split, rng, held)
        if ex.prompt in seen:
            continue
        seen.add(ex.prompt)
        out.append(ex)
    rng.shuffle(out)
    return out


def summarise(splits: dict) -> dict:
    """Corpus statistics for the report: sizes, family and attribute coverage.

    Args:
        splits: ``{split_name: [Example, ...]}``.

    Returns:
        A JSON-serialisable dict of per-split counts and cross-split checks.
    """
    stats = {"splits": {}}
    for name, rows in splits.items():
        fam = Counter(r.family for r in rows)
        stats["splits"][name] = {
            "n": len(rows),
            "families": dict(sorted(fam.items())),
            "attributes": dict(sorted(Counter(
                r.meta["attribute"] for r in rows).items())),
            "hops": dict(sorted(Counter(
                str(r.meta["hops"]) for r in rows).items())),
            "answers_distinct": len({r.answer for r in rows}),
            "held_out_phrasing": sum(r.meta["held_out_phrasing"] for r in rows),
            "mean_prompt_chars": round(
                sum(len(r.prompt) for r in rows) / max(len(rows), 1), 1),
        }
    # The two leakage checks, computed rather than asserted in prose.
    ents = {k: {e for r in v for e in r.meta["entities"]}
            for k, v in splits.items()}
    prompts = {k: {r.prompt for r in v} for k, v in splits.items()}
    stats["leakage"] = {
        "train_test_shared_entities": sorted(ents["train"] & ents["test"]),
        "train_test_shared_prompts": len(prompts["train"] & prompts["test"]),
        "val_test_shared_prompts": len(prompts["val"] & prompts["test"]),
        "train_val_shared_prompts": len(prompts["train"] & prompts["val"]),
    }
    return stats


def write_jsonl(path: Path, rows: list):
    """Write examples as one JSON object per line, UTF-8, unescaped."""
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({
                "family": r.family, "prompt": r.prompt, "answer": r.answer,
                "answer_direct": r.answer_direct,
                "answer_rationale": r.answer_rationale, "meta": r.meta,
            }, ensure_ascii=False) + "\n")


def write_sample_md(path: Path, language: str, splits: dict, per_family: int):
    """Write a small human-readable sample, one section per family.

    The full corpus goes to Drive (it is ~46 MB per language, mostly metadata),
    but "if it is not in the branch, it is not graded" -- so this committed
    excerpt is what a grader actually reads to see what the data looks like.

    Args:
        path: Output Markdown path.
        language: Language name, for the heading.
        splits: ``{split_name: [Example, ...]}``.
        per_family: Examples to show per family, drawn from the test split so
            the excerpt uses held-out entities.
    """
    rows = splits["test"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# Synthetic reasoning corpus - {language} (sample)\n\n"
                 f"Drawn from `test.jsonl`, so every name below is from the "
                 f"held-out entity pool.\nFull corpus on Google Drive; "
                 f"regenerate with `python -m src.reasoning.generate "
                 f"--language {language} ...`.\n\n")
        for fam in FAMILIES:
            spec = FAMILIES[fam]
            fh.write(f"## `{fam}` - {spec['n']} entities, {spec['premise']} "
                     f"premises, {spec['hops']} transitive hop"
                     f"{'s' if spec['hops'] != 1 else ''}\n\n")
            shown = [r for r in rows if r.family == fam][:per_family]
            for r in shown:
                fh.write(f"```\n{r.prompt.rstrip()}\n{r.answer_rationale}\n```\n")
                fh.write(f"*attribute* `{r.meta['attribute']}` - "
                         f"*phrasing* `{r.meta['premise_form']}`/"
                         f"`{r.meta['question_form']}`"
                         + (" - **held-out phrasing**"
                            if r.meta["held_out_phrasing"] else "") + "\n\n")


def tokenizer_stats(splits: dict, model_path: str) -> dict:
    """Measure how the language's own tokenizer segments this corpus.

    Numeral fragmentation is the reason this exists. Neither Phase 1 tokenizer
    merged multi-digit ASCII runs, but Hindi learned 42 Devanagari-numeral
    pieces against Nepali's 158, so the same three-digit quantity can cost a
    different number of tokens in the two models. If the value-premise families
    turn out to behave differently across H and L, this is the first place to
    look for why.

    Args:
        splits: ``{split_name: [Example, ...]}``.
        model_path: SentencePiece model for this language.

    Returns:
        Per-split token-length statistics, or ``{"available": False}`` when
        sentencepiece is not installed.
    """
    try:
        import sentencepiece as spm
    except ImportError:
        return {"available": False,
                "note": "sentencepiece not installed; rerun to populate"}
    sp = spm.SentencePieceProcessor(model_file=model_path)
    out = {"available": True, "model": model_path}
    for name, rows in splits.items():
        plen = [len(sp.encode(r.prompt)) for r in rows]
        alen = [len(sp.encode(r.answer_direct)) for r in rows]
        rlen = [len(sp.encode(r.answer_rationale)) for r in rows]
        # Cost of the quantities alone, on the families that state them.
        numerals = [len(sp.encode(templates.devanagari(v)))
                    for r in rows for v in r.meta["values"]
                    if r.meta["premise_form"] in ("plain", "colon")
                    and r.family.startswith("val")]
        out[name] = {
            "mean_prompt_tokens": round(sum(plen) / len(plen), 2),
            "max_prompt_tokens": max(plen),
            "mean_direct_answer_tokens": round(sum(alen) / len(alen), 2),
            "mean_rationale_answer_tokens": round(sum(rlen) / len(rlen), 2),
            "mean_numeral_tokens": (round(sum(numerals) / len(numerals), 3)
                                    if numerals else None),
        }
    return out


def main():
    """Parse the CLI and write one language's reasoning corpus."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--language", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train", type=int, default=40000)
    ap.add_argument("--val", type=int, default=5000)
    ap.add_argument("--test", type=int, default=5000)
    ap.add_argument("--held-out-phrasing-frac", type=float, default=0.25,
                    help="fraction of TEST examples using reserved phrasings")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenizer", default=None,
                    help="SentencePiece model, to record token-length stats")
    ap.add_argument("--sample-per-family", type=int, default=2)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Test first: it owns the deduplication set, so any collision with a later
    # train or val draw is resolved by discarding the train/val item.
    seen = set()
    splits = {}
    for name, n in (("test", args.test), ("val", args.val),
                    ("train", args.train)):
        splits[name] = build_split(args.language, name, n, rng, seen,
                                   args.held_out_phrasing_frac)
        write_jsonl(out / f"{name}.jsonl", splits[name])
        print(f"  {name:<5} {n:>7,} examples -> {out / (name + '.jsonl')}")

    write_sample_md(out / "sample.md", args.language, splits,
                    args.sample_per_family)
    stats = summarise(splits)
    tok = args.tokenizer or f"tokenizers/{args.language}/{args.language}_bpe_8k.model"
    if Path(tok).exists():
        stats["tokenizer"] = tokenizer_stats(splits, tok)
    stats["provenance"] = provenance.build(
        "src.reasoning.generate", inputs={},
        extra={"language": args.language, "seed": args.seed,
               "held_out_phrasing_frac": args.held_out_phrasing_frac})
    with open(out / "stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    leak = stats["leakage"]
    bad = (leak["train_test_shared_entities"] or leak["train_test_shared_prompts"]
           or leak["val_test_shared_prompts"] or leak["train_val_shared_prompts"])
    print(f"\n  leakage: entities {len(leak['train_test_shared_entities'])}, "
          f"prompts train/test {leak['train_test_shared_prompts']}, "
          f"val/test {leak['val_test_shared_prompts']}, "
          f"train/val {leak['train_val_shared_prompts']}")
    if bad:
        print("  LEAKAGE DETECTED", file=sys.stderr)
        return 1
    print(f"  stats -> {out / 'stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
