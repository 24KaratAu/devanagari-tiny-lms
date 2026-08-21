"""Intrinsic evaluation of a trained tokenizer on held-out text.

Produces every statistic Phase 1 §1.3 asks for — vocabulary size,
token-frequency statistics, average characters per token, tokenization
examples, and unknown-token statistics — plus a vocabulary-size sweep that
justifies the vocabulary choice on fertility rather than by assertion.
Use ``--language`` to select which example sentences are reported.

Comparison tokenizers are fitted on a sample of the *train* split and evaluated
on a sample of the *validation* split, so no model is scored on text it saw.

Usage:
    python scripts/benchmark_tokenizer.py \
        --model  <path>/hindi_bpe_8k.model \
        --val    <path>/val_sample.txt \
        --train  <path>/train_sample.txt \
        --out    report/hindi_tokenizer_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from collections import Counter
from pathlib import Path

import sentencepiece as spm

# Devanagari dependent vowel signs, anusvara/visarga/candrabindu and vocalic
# marks. A token that *begins* with one of these is a broken grapheme cluster:
# the mark has been split away from the consonant it belongs to.
COMBINING = (set(chr(c) for c in range(0x0900, 0x0904))
             | set(chr(c) for c in range(0x093A, 0x0940))
             | set(chr(c) for c in range(0x0940, 0x0950))
             | {chr(0x0962), chr(0x0963), chr(0x094D)})

WORD_RE = re.compile(r"\S+")


def load_docs(path: Path, limit_chars: int) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")[:limit_chars]
    return [ln.strip() for ln in text.split("\n") if len(ln.strip()) > 20]


def evaluate(sp: spm.SentencePieceProcessor, docs: list[str]) -> dict:
    tokens = words = chars = broken = unk = 0
    freq: Counter[int] = Counter()
    unk_id = sp.unk_id()
    for doc in docs:
        ids = sp.encode(doc)
        pieces = sp.encode(doc, out_type=str)
        tokens += len(ids)
        words += len(WORD_RE.findall(doc))
        chars += len(doc)
        freq.update(ids)
        for piece in pieces:
            bare = piece.lstrip("▁")
            if bare and bare[0] in COMBINING:
                broken += 1
        unk += sum(1 for i in ids if i == unk_id)

    total = sum(freq.values())
    probs = [c / total for c in freq.values()]
    shannon = -sum(p * math.log2(p) for p in probs)
    renyi2 = -math.log2(sum(p * p for p in probs))
    return {
        "documents": len(docs),
        "characters": chars,
        "words": words,
        "tokens": tokens,
        "fertility_tokens_per_word": round(tokens / words, 4),
        "characters_per_token": round(chars / tokens, 4),
        "broken_grapheme_rate_pct": round(100 * broken / tokens, 4),
        "unknown_tokens": unk,
        "unknown_token_rate_pct": round(100 * unk / tokens, 6),
        "vocab_size": sp.get_piece_size(),
        "vocab_types_observed": len(freq),
        "vocab_coverage_pct": round(100 * len(freq) / sp.get_piece_size(), 2),
        "shannon_entropy_bits": round(shannon, 4),
        "renyi_entropy_alpha2_bits": round(renyi2, 4),
        "_freq": freq,
    }


def train_variant(train_file: Path, vocab_size: int, model_type: str,
                  workdir: Path, input_sentence_size: int = 200_000
                  ) -> spm.SentencePieceProcessor:
    prefix = workdir / f"{model_type}_{vocab_size}"
    spm.SentencePieceTrainer.train(
        input=str(train_file), model_prefix=str(prefix), vocab_size=vocab_size,
        model_type=model_type, character_coverage=0.9995, num_threads=8,
        max_sentence_length=8192, input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
        byte_fallback=True, split_by_unicode_script=True,
        split_by_whitespace=True, split_digits=True,
    )
    sp = spm.SentencePieceProcessor()
    sp.load(f"{prefix}.model")
    return sp


SAMPLES = {
    "hindi": [
        "भारत एक विशाल और विविधतापूर्ण दक्षिण एशियाई गणराज्य है।",
        "विश्वविद्यालय के विद्यार्थियों ने वैज्ञानिक अनुसंधान में महत्वपूर्ण योगदान दिया।",
        "प्रधानमंत्री ने कहा कि अर्थव्यवस्था तेज़ी से आगे बढ़ रही है।",
        "उन्होंने 2026 में लगभग 15 प्रतिशत की वृद्धि दर्ज की।",
    ],
    "nepali": [
        "नेपाल एक सुन्दर र सार्वभौमसत्ता सम्पन्न दक्षिण एसियाली राष्ट्र हो।",
        "विश्वविद्यालयमा विद्यार्थीहरूले वैज्ञानिक अनुसन्धान गरिरहेका छन्।",
        "प्रधानमन्त्रीले अर्थतन्त्र तीव्र गतिमा अगाडि बढिरहेको बताउनुभयो।",
        "उनीहरूले २०२६ मा करिब १५ प्रतिशत वृद्धि दर्ज गरे।",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default="report/hindi_tokenizer_benchmark.json")
    ap.add_argument("--val-chars", type=int, default=20_000_000)
    ap.add_argument("--sweep", default="4000,8000,16000,32000")
    ap.add_argument("--skip-sweep", action="store_true",
                    help="Only evaluate --model; do not fit comparison tokenizers.")
    ap.add_argument("--algorithms", default=None,
                    help="Head-to-head at one vocab size, e.g. 'bpe,unigram'. Fits each "
                         "on identical data with identical settings, which is what makes "
                         "the algorithm comparison fair (report 6.2.2).")
    ap.add_argument("--algorithms-vocab", type=int, default=8000)
    ap.add_argument("--input-sentence-size", type=int, default=200_000,
                    help="SentencePiece samples at most this many lines. The shipped "
                         "tokenizers used 1,000,000; match it for a like-for-like test.")
    ap.add_argument("--language", choices=sorted(SAMPLES), default="hindi",
                    help="Selects the example sentences shown in the report.")
    args = ap.parse_args()

    val_docs = load_docs(Path(args.val), args.val_chars)
    print(f"Held-out evaluation set: {len(val_docs):,} documents", flush=True)

    sp = spm.SentencePieceProcessor()
    sp.load(args.model)
    prod = evaluate(sp, val_docs)
    freq = prod.pop("_freq")

    # --- token-frequency statistics ---
    ranked = freq.most_common()
    total = sum(freq.values())
    cum, coverage = 0, {}
    for k in (100, 500, 1000, 2000, 4000):
        cum = sum(c for _, c in ranked[:k])
        coverage[f"top_{k}_pct_of_tokens"] = round(100 * cum / total, 2)
    singletons = sum(1 for _, c in ranked if c == 1)
    top_pieces = [{"piece": sp.id_to_piece(i), "count": c,
                   "pct": round(100 * c / total, 4)} for i, c in ranked[:25]]
    prod["token_frequency"] = {
        "distinct_types_observed": len(freq),
        "types_seen_once": singletons,
        "coverage": coverage,
        "top_25_pieces": top_pieces,
    }

    # --- tokenization examples ---
    prod["examples"] = [{"text": s,
                         "pieces": sp.encode(s, out_type=str),
                         "n_tokens": len(sp.encode(s)),
                         "n_words": len(WORD_RE.findall(s))}
                        for s in SAMPLES[args.language]]

    # --- vocabulary-size sweep + algorithm comparison ---
    sweep = []
    if args.skip_sweep:
        print("  (sweep skipped)", flush=True)
    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        train_file = Path(args.train)
        if args.algorithms:
            for mt in [x.strip() for x in args.algorithms.split(",")]:
                print(f"  fitting {mt} {args.algorithms_vocab} on "
                      f"{args.input_sentence_size:,} sentences...", flush=True)
                v = evaluate(train_variant(train_file, args.algorithms_vocab, mt,
                                           workdir, args.input_sentence_size), val_docs)
                v.pop("_freq")
                v["model_type"] = mt
                v["requested_vocab"] = args.algorithms_vocab
                v["input_sentence_size"] = args.input_sentence_size
                sweep.append(v)
        for vs in ([] if (args.skip_sweep or args.algorithms)
                   else [int(v) for v in args.sweep.split(",")]):
            for mt in (("bpe", "unigram") if vs == 8000 else ("bpe",)):
                print(f"  fitting {mt} {vs}...", flush=True)
                v = evaluate(train_variant(train_file, vs, mt, workdir), val_docs)
                v.pop("_freq")
                v["model_type"], v["requested_vocab"] = mt, vs
                sweep.append(v)
    prod["sweep"] = sweep

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(prod, indent=2, ensure_ascii=False))

    print(f"\nProduction {Path(args.model).stem} on {prod['documents']:,} held-out docs")
    print(f"  vocabulary size          {prod['vocab_size']:,}")
    print(f"  fertility (tok/word)     {prod['fertility_tokens_per_word']}")
    print(f"  characters / token       {prod['characters_per_token']}")
    print(f"  unknown-token rate       {prod['unknown_token_rate_pct']}%")
    print(f"  broken grapheme rate     {prod['broken_grapheme_rate_pct']}%")
    print(f"  vocabulary coverage      {prod['vocab_coverage_pct']}%")
    print(f"  Shannon / Renyi-2 (bits) {prod['shannon_entropy_bits']} / "
          f"{prod['renyi_entropy_alpha2_bits']}")
    print(f"\n{'model':>10}{'vocab':>8}{'fertility':>11}{'chars/tok':>11}"
          f"{'unk%':>8}{'broken%':>9}{'coverage%':>11}")
    for v in sweep:
        print(f"{v['model_type']:>10}{v['requested_vocab']:>8}"
              f"{v['fertility_tokens_per_word']:>11}{v['characters_per_token']:>11}"
              f"{v['unknown_token_rate_pct']:>8}{v['broken_grapheme_rate_pct']:>9}"
              f"{v['vocab_coverage_pct']:>11}")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
