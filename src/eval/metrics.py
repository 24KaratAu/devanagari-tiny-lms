"""Generation-quality metrics: BLEU-4, chrF++, ROUGE-L, repetition, Distinct-n.

Brief section 2.4 asks for these five families over the generated continuations,
plus a discussion of "why each metric is or is not informative for this setting".
Every metric here is implemented from the definition rather than pulled from a
library, for three reasons: the brief's guideline 2 already rules out borrowed
model code and the same spirit applies to the numbers we report; the standard
implementations disagree on tokenisation for Devanagari; and a metric we wrote
is a metric we can explain in the report.

Reading the numbers. These prompts are *open-ended continuations*, not
translations or summaries, so there is exactly one reference and no reason the
model should reproduce it. BLEU-4 and ROUGE-L therefore measure something close
to "did the model guess the same words the corpus happened to use next", which
for a 25M-parameter LM is near zero and stays near zero however good the text
is. They are reported because the brief requires them and because their
*ordering* across decoding settings is still informative -- greedy scores
highest not because it writes better text but because it hedges towards frequent
words. chrF++ degrades more gracefully: character n-grams give partial credit
for the right stem under the wrong inflection, which matters in two heavily
inflected languages. The reference-free metrics -- repetition rate and
Distinct-1/2 -- are the ones that actually separate the decoding settings, and
each is reported against the held-out human text as a control, so "greedy
repeats 4-grams 12x more than real Nepali does" is a statement with a baseline.

Tokenisation. Words are maximal non-whitespace runs with leading and trailing
punctuation split off. Python's ``\\w`` is not usable here: Devanagari vowel
signs are combining marks (category Mn) and are not alphanumeric, so ``\\w+``
would tear the matras off their consonants. Character n-grams follow the chrF
convention and are computed with whitespace removed.

Cross-model caution. Model H and Model L are scored on different corpora in
different languages, so an absolute BLEU gap between them means little. The
comparisons that carry weight are within a model across decoding settings, and
between a model and its own reference control.

Usage:
    python -m src.eval.metrics \
        --generations hindi=hindi/samples/generations.json \
                      nepali=nepali/samples/generations.json \
        --out report/figures/generation_metrics.json \
        --markdown report/figures/generation_metrics.md
"""

import argparse
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path

from src.eval import provenance

MAX_BLEU_ORDER = 4


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #

def _is_punct(ch: str) -> bool:
    """True for Unicode punctuation and symbols, including the Devanagari danda."""
    return unicodedata.category(ch)[0] in ("P", "S")


def word_tokenize(text: str) -> list:
    """Split text into word and punctuation tokens.

    Whitespace-delimited chunks are kept whole except that runs of punctuation
    at either end become their own tokens, so ``"दिल्ली।"`` yields ``["दिल्ली", "।"]``
    while the internal combining marks of ``"दिल्ली"`` stay attached.

    Args:
        text: Raw generated or reference text.

    Returns:
        A list of tokens.
    """
    out = []
    for chunk in text.split():
        i, j = 0, len(chunk)
        while i < j and _is_punct(chunk[i]):
            out.append(chunk[i])
            i += 1
        tail = []
        while j > i and _is_punct(chunk[j - 1]):
            tail.append(chunk[j - 1])
            j -= 1
        if i < j:
            out.append(chunk[i:j])
        out.extend(reversed(tail))
    return out


def char_sequence(text: str) -> str:
    """Text with all whitespace removed, as chrF's character n-grams require."""
    return "".join(text.split())


def ngrams(seq, n: int) -> Counter:
    """Count the ``n``-grams of a sequence (a list of tokens or a string)."""
    return Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))


# --------------------------------------------------------------------------- #
# BLEU
# --------------------------------------------------------------------------- #

def corpus_bleu(hyps: list, refs: list, max_order: int = MAX_BLEU_ORDER) -> dict:
    """Corpus-level BLEU with a single reference per hypothesis.

    Modified n-gram precisions are pooled over the whole corpus -- clipped match
    counts summed in the numerator, candidate n-gram counts in the denominator --
    rather than averaged per sentence, which is what "corpus BLEU" means and what
    keeps a single lucky short sample from dominating.

    Zero counts at some order are the norm on open-ended continuations, and an
    unsmoothed geometric mean would then collapse to exactly 0 and hide the
    ordering between decoding settings. The reported ``bleu`` therefore uses the
    standard exponential smoothing (each successive zero order is credited
    ``1 / (2^k * denominator)``); ``bleu_unsmoothed`` and the raw ``precisions``
    are reported alongside so nothing is hidden behind the smoothing.

    Args:
        hyps: Generated strings.
        refs: Reference strings, aligned with ``hyps``.
        max_order: Highest n-gram order, 4 for BLEU-4.

    Returns:
        BLEU (smoothed and not), per-order precisions, brevity penalty, and the
        length ratio.
    """
    matches = [0] * max_order
    totals = [0] * max_order
    hyp_len = ref_len = 0

    for h, r in zip(hyps, refs):
        ht, rt = word_tokenize(h), word_tokenize(r)
        hyp_len += len(ht)
        ref_len += len(rt)
        for n in range(1, max_order + 1):
            hc, rc = ngrams(ht, n), ngrams(rt, n)
            # Clipping: a candidate n-gram can match at most as often as it
            # appears in the reference, which is what stops "the the the the".
            matches[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            totals[n - 1] += sum(hc.values())

    precisions, smoothed, k = [], [], 0
    for n in range(max_order):
        p = matches[n] / totals[n] if totals[n] else 0.0
        precisions.append(p)
        if matches[n] == 0:
            k += 1
            smoothed.append(1.0 / (2 ** k * totals[n]) if totals[n] else 0.0)
        else:
            smoothed.append(p)

    bp = 1.0 if hyp_len > ref_len else (
        math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0)

    def _geo(ps):
        if any(p <= 0 for p in ps):
            return 0.0
        return math.exp(sum(math.log(p) for p in ps) / len(ps))

    return {
        "bleu": 100.0 * bp * _geo(smoothed),
        "bleu_unsmoothed": 100.0 * bp * _geo(precisions),
        "precisions": [100.0 * p for p in precisions],
        "brevity_penalty": bp,
        "length_ratio": hyp_len / ref_len if ref_len else 0.0,
        "hyp_len": hyp_len, "ref_len": ref_len,
    }


# --------------------------------------------------------------------------- #
# chrF++
# --------------------------------------------------------------------------- #

def chrf_plus(hyps: list, refs: list, char_order: int = 6,
              word_order: int = 2, beta: float = 2.0) -> dict:
    """Corpus-level chrF++ -- character n-grams 1..6 plus word n-grams 1..2.

    chrF averages the per-order F-scores of character n-grams; chrF++ extends
    the average with short word n-grams, which restores a little word-order
    sensitivity that pure character matching loses. ``beta=2`` weights recall
    twice as heavily as precision, the published default.

    Statistics are pooled across the corpus before the F-score is taken, so the
    result is a corpus metric rather than a mean of sentence scores. Orders whose
    denominators are empty everywhere (a 6-gram order on very short text) are
    dropped from the average instead of contributing a spurious zero.

    Args:
        hyps: Generated strings.
        refs: Reference strings, aligned with ``hyps``.
        char_order: Highest character n-gram order.
        word_order: Highest word n-gram order (0 gives plain chrF).
        beta: Recall weight.

    Returns:
        chrF++ overall plus the character-only and word-only components.
    """
    def _stats(order: int, seq_fn):
        match = hyp_tot = ref_tot = 0
        for h, r in zip(hyps, refs):
            hc, rc = ngrams(seq_fn(h), order), ngrams(seq_fn(r), order)
            match += sum(min(c, rc[g]) for g, c in hc.items())
            hyp_tot += sum(hc.values())
            ref_tot += sum(rc.values())
        return match, hyp_tot, ref_tot

    def _f(match, hyp_tot, ref_tot):
        if hyp_tot == 0 or ref_tot == 0:
            return None
        p, r = match / hyp_tot, match / ref_tot
        if p + r == 0:
            return 0.0
        b2 = beta ** 2
        return (1 + b2) * p * r / (b2 * p + r)

    char_f = [f for n in range(1, char_order + 1)
              if (f := _f(*_stats(n, char_sequence))) is not None]
    word_f = [f for n in range(1, word_order + 1)
              if (f := _f(*_stats(n, word_tokenize))) is not None]

    combined = char_f + word_f
    mean = lambda xs: 100.0 * sum(xs) / len(xs) if xs else 0.0
    return {"chrf_plus": mean(combined), "chrf": mean(char_f),
            "word_f": mean(word_f), "char_orders": len(char_f),
            "word_orders": len(word_f)}


# --------------------------------------------------------------------------- #
# ROUGE-L
# --------------------------------------------------------------------------- #

def lcs_length(a: list, b: list) -> int:
    """Length of the longest common subsequence, by dynamic programming.

    Only two rows of the table are kept, so memory is O(min(len)) rather than
    O(len^2); the continuations here are ~128 tokens, but the same call is used
    on references that can be longer.
    """
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(hyps: list, refs: list, beta: float = 1.0) -> dict:
    """ROUGE-L: LCS-based F-measure, averaged over samples.

    ROUGE is conventionally reported as the mean of per-sample scores, not as a
    corpus-pooled statistic, so that is what ``rouge_l_f`` is. Unlike BLEU it
    rewards in-order matches that are not contiguous, which is a slightly better
    fit for free continuation -- though it still assumes the reference is the
    target, which here it is not.

    Args:
        hyps: Generated strings.
        refs: Reference strings, aligned with ``hyps``.
        beta: Recall weight; 1.0 gives the F1 usually reported.

    Returns:
        Mean F, precision and recall.
    """
    fs, ps, rs = [], [], []
    for h, r in zip(hyps, refs):
        ht, rt = word_tokenize(h), word_tokenize(r)
        l = lcs_length(ht, rt)
        p = l / len(ht) if ht else 0.0
        rec = l / len(rt) if rt else 0.0
        b2 = beta ** 2
        f = 0.0 if p + rec == 0 else (1 + b2) * p * rec / (b2 * p + rec)
        fs.append(f); ps.append(p); rs.append(rec)

    mean = lambda xs: 100.0 * sum(xs) / len(xs) if xs else 0.0
    return {"rouge_l_f": mean(fs), "rouge_l_precision": mean(ps),
            "rouge_l_recall": mean(rs)}


# --------------------------------------------------------------------------- #
# Reference-free: repetition and diversity
# --------------------------------------------------------------------------- #

def repetition_and_diversity(texts: list, max_n: int = 4) -> dict:
    """Repetition rate and Distinct-n over a set of generations.

    Two different questions, deliberately kept apart:

      * ``rep_n`` -- *within* one sample, the fraction of its n-grams that are
        repeats of an n-gram it already produced, averaged over samples. This is
        the degeneracy that greedy decoding falls into, and it is invisible to
        BLEU because a repeated phrase can still contain reference n-grams.
      * ``distinct_n`` -- *across* the whole set, unique n-grams over total
        n-grams. This measures whether every prompt is being answered with the
        same generic text.

    Distinct-n falls as the sample count and length grow, so it is only
    comparable between sets of the same size and length -- which is exactly how
    it is used here, since every decoding setting generates the same number of
    continuations of the same requested length from the same prompts.

    Args:
        texts: Generated (or reference) strings.
        max_n: Highest n-gram order for the repetition rates.

    Returns:
        ``rep_1..rep_n``, ``distinct_1``, ``distinct_2``, and length statistics.
    """
    per_sample_rep = {n: [] for n in range(1, max_n + 1)}
    corpus_counts = {1: Counter(), 2: Counter()}
    corpus_totals = {1: 0, 2: 0}
    lengths = []

    for t in texts:
        toks = word_tokenize(t)
        lengths.append(len(toks))
        for n in range(1, max_n + 1):
            c = ngrams(toks, n)
            total = sum(c.values())
            if total:
                # 1 - unique/total is the share of n-grams that are repeats.
                per_sample_rep[n].append(1.0 - len(c) / total)
        for n in (1, 2):
            c = ngrams(toks, n)
            corpus_counts[n].update(c)
            corpus_totals[n] += sum(c.values())

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    out = {f"rep_{n}": 100.0 * mean(per_sample_rep[n]) for n in range(1, max_n + 1)}
    for n in (1, 2):
        out[f"distinct_{n}"] = (100.0 * len(corpus_counts[n]) / corpus_totals[n]
                               if corpus_totals[n] else 0.0)
    out["mean_words"] = mean(lengths)
    out["n_samples"] = len(texts)
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def score_setting(hyps: list, refs: list) -> dict:
    """Every metric for one decoding setting."""
    out = {}
    out.update(corpus_bleu(hyps, refs))
    out.update(chrf_plus(hyps, refs))
    out.update(rouge_l(hyps, refs))
    out.update(repetition_and_diversity(hyps))
    return out


def score_generations(payload: dict) -> dict:
    """Score every decoding setting in one ``generations.json``.

    A ``reference`` row is added from the true held-out continuations. It has no
    BLEU or ROUGE against itself -- those would be 100 by construction -- but its
    repetition and Distinct-n figures are the human baseline the generations are
    read against.

    Args:
        payload: The dict written by :mod:`src.eval.generate`.

    Returns:
        ``{"language", "settings": {name: metrics}, "reference": metrics}``.
    """
    records = payload["records"]
    refs = [r["reference_text"] for r in records]

    settings = {}
    for name in payload["settings"]:
        hyps = [r["generations"][name]["text"] for r in records]
        m = score_setting(hyps, refs)
        gens = [r["generations"][name] for r in records]
        m["mean_tokens"] = sum(g["n_tokens"] for g in gens) / len(gens)
        m["eos_rate"] = 100.0 * sum(g["hit_eos"] for g in gens) / len(gens)
        settings[name] = m

    return {"language": payload["language"],
            "checkpoint_step": payload.get("checkpoint_step"),
            "n_prompts": len(records),
            "prompt_tokens": payload.get("prompt_tokens"),
            "max_new_tokens": payload.get("max_new_tokens"),
            "settings": settings,
            "reference": repetition_and_diversity(refs)}


def markdown_table(results: dict) -> str:
    """Render the scored settings as the report's generation-metrics table."""
    cols = [("BLEU-4", "bleu", "{:.2f}"), ("chrF++", "chrf_plus", "{:.2f}"),
            ("ROUGE-L", "rouge_l_f", "{:.2f}"), ("rep-1", "rep_1", "{:.1f}"),
            ("rep-4", "rep_4", "{:.1f}"), ("Distinct-1", "distinct_1", "{:.1f}"),
            ("Distinct-2", "distinct_2", "{:.1f}"),
            ("<eos> %", "eos_rate", "{:.0f}"), ("words", "mean_words", "{:.0f}")]

    lines = []
    for lang, res in results.items():
        lines += [f"### {lang} ({res['n_prompts']} prompts, "
                  f"{res['prompt_tokens']}-token prefix, "
                  f"{res['max_new_tokens']} new tokens)", "",
                  "| Decoding | " + " | ".join(c[0] for c in cols) + " |",
                  "|:--|" + "--:|" * len(cols)]
        for name, m in res["settings"].items():
            cells = [c[2].format(m.get(c[1], float("nan"))) for c in cols]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        ref = res["reference"]
        cells = ["--", "--", "--",
                 f"{ref['rep_1']:.1f}", f"{ref['rep_4']:.1f}",
                 f"{ref['distinct_1']:.1f}", f"{ref['distinct_2']:.1f}",
                 "--", f"{ref['mean_words']:.0f}"]
        lines.append("| *held-out reference* | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", nargs="+", required=True,
                    help="lang=path/to/generations.json, repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    results, sources = {}, {}
    for spec in args.generations:
        lang, path = spec.split("=", 1)
        sources[lang] = path
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        res = score_generations(payload)
        # Carry the generating run's provenance through, so the metrics point
        # back to the checkpoint that produced the text and not merely to the
        # JSON in between.
        res["generation_provenance"] = payload.get("provenance")
        results[lang] = res
        print(f"\n{lang}: {res['n_prompts']} prompts")
        for name, m in res["settings"].items():
            print(f"  {name:10} BLEU {m['bleu']:5.2f}  chrF++ {m['chrf_plus']:5.2f}  "
                  f"ROUGE-L {m['rouge_l_f']:5.2f}  rep-4 {m['rep_4']:5.1f}%  "
                  f"D-1 {m['distinct_1']:5.1f}%  D-2 {m['distinct_2']:5.1f}%")
        r = res["reference"]
        print(f"  {'reference':10} {'':29}{'':16}rep-4 {r['rep_4']:5.1f}%  "
              f"D-1 {r['distinct_1']:5.1f}%  D-2 {r['distinct_2']:5.1f}%")

    payload = {"provenance": provenance.build("src.eval.metrics", sources),
               "languages": results}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")

    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(markdown_table(results), encoding="utf-8")
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
