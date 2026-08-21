"""Reproduce every measured claim the Phase 1 report makes about the corpora.

Each subcommand regenerates one set of figures that appears in
`report/phase1_report.md`, so no number in the report rests on a throwaway
script. Shards are streamed from Drive with `rclone cat` and byte-verified
against the size Drive reports, because a truncated stream through a pipe is
indistinguishable from a short file (report §8.1, failure 3).

    uniqueness     §4.3, §5.1  per-source unique documents and characters
    shard-overlap  §4.3        pairwise shard hash/URL intersection
    cross-corpus   §4.3        the two corpora share no documents
    nfc            §4.1        Unicode NFC verification
    numerals       §6.1        ASCII vs Devanagari digit usage, 0-100 encoding
    token-share    §5.3        manual vs automated token share under a tokenizer

Examples:
    python scripts/verify_corpus_claims.py uniqueness --language nepali
    python scripts/verify_corpus_claims.py cross-corpus
    python scripts/verify_corpus_claims.py numerals --language nepali \
        --tokenizer tokenizers/nepali/nepali_bpe_8k.model
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import unicodedata
from collections import defaultdict

DRIVE = "gdrive:24KaratAu"
ASCII_DIGITS = set("0123456789")
DEV_DIGITS = set("०१२३४५६७८९")


# ---------------------------------------------------------------- plumbing

def lsjson(remote: str) -> list[dict]:
    out = subprocess.run(["rclone", "lsjson", remote],
                         capture_output=True, text=True, check=True).stdout
    return sorted((e for e in json.loads(out) if e["Name"].endswith(".jsonl")),
                  key=lambda e: e["Name"])


def stream_shard(remote: str, name: str, expected: int, attempts: int = 3):
    """Yield decoded docs, verifying the byte count against Drive's size."""
    for attempt in range(attempts):
        proc = subprocess.Popen(["rclone", "cat", f"{remote}/{name}",
                                 "--retries", "5", "--timeout", "60s"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        docs, read = [], 0
        for raw in proc.stdout:
            read += len(raw)
            if not raw.strip():
                continue
            try:
                docs.append(json.loads(raw))
            except Exception:
                pass
        proc.stdout.close()
        proc.wait()
        if read == expected:
            return docs
        print(f"    {name}: short read {read:,} != {expected:,} "
              f"(attempt {attempt + 1}/{attempts})", file=sys.stderr)
    raise RuntimeError(f"{name}: incomplete after {attempts} attempts")


def source_group(name: str) -> str:
    if name.startswith("wiki_shard"):
        return "Wikipedia dump"
    if name.startswith("manual_web"):
        return "Web crawl (sitemap)"
    if name.startswith("manual_news"):
        return "News crawl (legacy)"
    if name.startswith("manual_aux"):
        return "Wikibooks / Wikiquote"
    if name.startswith("manual_lit"):
        return "Literature re-parse"
    if name.startswith("wikisource"):
        return "Wikisource"
    return "Other"


def load_tokenizer(path: str):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp


# ---------------------------------------------------------------- commands

def cmd_uniqueness(a) -> int:
    """§4.3, §5.1 — how much of the manual partition is unique, by source."""
    remote = f"{DRIVE}/{a.language}/cleaned/manual"
    files = lsjson(remote)
    sp = load_tokenizer(a.tokenizer) if a.tokenizer else None
    rng = random.Random(7)
    seen: set[str] = set()
    g = defaultdict(lambda: {"d": 0, "c": 0, "ud": 0, "uc": 0})
    tot = uni = totc = unic = 0
    s_chars = s_tokens = 0

    for e in files:
        key = source_group(e["Name"])
        for doc in stream_shard(remote, e["Name"], e["Size"]):
            text = doc.get("text", "")
            if not text.strip():
                continue
            tot += 1
            totc += len(text)
            g[key]["d"] += 1
            g[key]["c"] += len(text)
            h = (doc.get("text_sha256") or "")[:16] or text[:200]
            if h in seen:
                continue
            seen.add(h)
            uni += 1
            unic += len(text)
            g[key]["ud"] += 1
            g[key]["uc"] += len(text)
            if sp is not None and rng.random() < a.sample_rate:
                s_chars += len(text)
                s_tokens += len(sp.encode(text))
        print(f"  {e['Name']}: unique {uni:,} docs / {unic:,} chars",
              file=sys.stderr, flush=True)

    print(f"\n{a.language.upper()} manual partition ({len(files)} shards)")
    print(f"{'Source':<24}{'docs uniq/all':>22}{'chars uniq/all':>30}{'uniq%':>8}")
    for k, v in sorted(g.items(), key=lambda kv: -kv[1]["c"]):
        pct = 100 * v["uc"] / v["c"] if v["c"] else 0
        print(f"{k:<24}{v['ud']:>10,}/{v['d']:<11,}{v['uc']:>14,}/{v['c']:<15,}{pct:>7.1f}%")
    print(f"{'TOTAL':<24}{uni:>10,}/{tot:<11,}{unic:>14,}/{totc:<15,}"
          f"{100 * unic / totc:>7.1f}%")
    if s_tokens:
        cpt = s_chars / s_tokens
        print(f"\nchars/token {cpt:.3f} -> unique manual tokens {unic / cpt:,.0f}")
    return 0


def cmd_shard_overlap(a) -> int:
    """§4.3 — same text under different URLs, which URL dedup cannot see."""
    remote = f"{DRIVE}/{a.language}/cleaned/manual"
    sizes = {e["Name"]: e["Size"] for e in lsjson(remote)}
    data = {}
    for n in a.shards:
        docs = stream_shard(remote, n, sizes[n])
        data[n] = ({(d.get("text_sha256") or "")[:16] for d in docs if d.get("text_sha256")},
                   {d["url"] for d in docs if d.get("url")})
        print(f"  {n}: {len(docs):,} docs, {len(data[n][0]):,} hashes, "
              f"{len(data[n][1]):,} urls", file=sys.stderr)
    print(f"\n{'pair':<52}{'shared hashes':>16}{'shared urls':>14}")
    names = list(data)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            (ha, ua), (hb, ub) = data[names[i]], data[names[j]]
            pct = 100 * len(ha & hb) / max(len(ha), 1)
            print(f"{names[i]} n {names[j]:<22}{len(ha & hb):>10,} ({pct:4.1f}%)"
                  f"{len(ua & ub):>14,}")
    return 0


def cmd_cross_corpus(a) -> int:
    """§4.3 — the two corpora must share no documents."""
    def sample(lang, sub, names, cap):
        remote = f"{DRIVE}/{lang}/cleaned/{sub}"
        sizes = {e["Name"]: e["Size"] for e in lsjson(remote)}
        out = set()
        for n in names:
            if n not in sizes:
                continue
            for d in stream_shard(remote, n, sizes[n])[:cap]:
                h = d.get("text_sha256")
                if h:
                    out.add(h[:16])
        return out

    pairs = [
        ("Sangraha verified/hin vs verified/nep", "automated_nonweb",
         ["nonweb_shard_0005.jsonl", "nonweb_shard_0100.jsonl"]),
        ("Wikipedia dumps hiwiki vs newiki", "manual",
         ["wiki_shard_0002.jsonl"]),
    ]
    print(f"{'comparison':<44}{'hindi':>10}{'nepali':>10}{'shared':>9}")
    for label, sub, names in pairs:
        h = sample("hindi", sub, names, a.cap)
        n = sample("nepali", sub, names, a.cap)
        print(f"{label:<44}{len(h):>10,}{len(n):>10,}{len(h & n):>9,}")
    return 0


def cmd_nfc(a) -> int:
    """§4.1 — Unicode normalisation is a verified no-op on this corpus."""
    remote = f"{DRIVE}/{a.language}/cleaned/manual"
    sizes = {e["Name"]: e["Size"] for e in lsjson(remote)}
    names = a.shards or list(sizes)[: a.shard_limit]
    docs = chars = nonnfc = 0
    for n in names:
        for d in stream_shard(remote, n, sizes[n]):
            t = d.get("text", "")
            if not t.strip():
                continue
            docs += 1
            chars += len(t)
            if unicodedata.normalize("NFC", t) != t:
                nonnfc += 1
    print(f"{a.language}: {docs:,} documents / {chars:,} characters")
    print(f"  non-NFC documents: {nonnfc:,} ({100 * nonnfc / max(docs, 1):.4f}%)")
    print(f"  -> NFC normalisation is {'a no-op' if not nonnfc else 'a correction'}")
    return 0


def cmd_numerals(a) -> int:
    """§6.1 — which numeral system each corpus uses, and how it encodes."""
    remote = f"{DRIVE}/{a.language}/cleaned/manual"
    sizes = {e["Name"]: e["Size"] for e in lsjson(remote)}
    names = a.shards or list(sizes)[: a.shard_limit]
    asc = dev = chars = 0
    for n in names:
        for d in stream_shard(remote, n, sizes[n]):
            t = d.get("text", "")
            chars += len(t)
            asc += sum(1 for c in t if c in ASCII_DIGITS)
            dev += sum(1 for c in t if c in DEV_DIGITS)
    tot = asc + dev
    print(f"{a.language}: {chars:,} characters sampled")
    print(f"  ASCII 0-9      {asc:>12,}  {100 * asc / max(tot, 1):5.1f}%")
    print(f"  Devanagari ०-९ {dev:>12,}  {100 * dev / max(tot, 1):5.1f}%")

    if a.tokenizer:
        sp = load_tokenizer(a.tokenizer)
        one = 0
        for v in range(101):
            dev_s = str(v).translate(str.maketrans("0123456789", "०१२३४५६७८९"))
            pieces = [p for p in sp.encode(dev_s, out_type=str) if p != "▁"]
            one += len(pieces) == 1
        print(f"\n  Devanagari values 0-100 encoding as a SINGLE token: {one}/101")
        for probe in ("2026", "२०२६", "१५", "६०"):
            print(f"    {probe:>6} -> {sp.encode(probe, out_type=str)}")
    return 0


def cmd_token_share(a) -> int:
    """§5.3 — manual vs automated token share under the shipped tokenizer."""
    sp = load_tokenizer(a.tokenizer)
    rng = random.Random(5)

    def cpt(sub, names):
        remote = f"{DRIVE}/{a.language}/cleaned/{sub}"
        sizes = {e["Name"]: e["Size"] for e in lsjson(remote)}
        c = t = 0
        for n in names:
            for d in stream_shard(remote, n, sizes[n]):
                x = d.get("text", "")
                if not x.strip() or rng.random() >= a.sample_rate:
                    continue
                c += len(x)
                t += len(sp.encode(x))
        return c / t

    m = cpt("manual", a.manual_shards)
    au = cpt("automated_nonweb", a.auto_shards)
    mt, at = a.manual_chars / m, a.auto_chars / au
    print(f"  manual    chars/token {m:.3f} -> {mt:,.0f} tokens")
    print(f"  automated chars/token {au:.3f} -> {at:,.0f} tokens")
    print(f"  total {mt + at:,.0f} tokens")
    print(f"  MANUAL TOKEN SHARE {mt / (mt + at) * 100:.2f}%  "
          f"(floor is 20%: {'PASS' if mt / (mt + at) >= 0.20 else 'FAIL'})")
    return 0


# ---------------------------------------------------------------- cli

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("uniqueness"); p.set_defaults(fn=cmd_uniqueness)
    p.add_argument("--language", default="nepali", choices=("hindi", "nepali"))
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--sample-rate", type=float, default=0.03)

    p = sub.add_parser("shard-overlap"); p.set_defaults(fn=cmd_shard_overlap)
    p.add_argument("--language", default="nepali", choices=("hindi", "nepali"))
    p.add_argument("--shards", nargs="+", default=["manual_news_shard_0020.jsonl",
                                                   "manual_news_shard_0030.jsonl",
                                                   "manual_news_shard_0040.jsonl"])

    p = sub.add_parser("cross-corpus"); p.set_defaults(fn=cmd_cross_corpus)
    p.add_argument("--cap", type=int, default=6000)

    p = sub.add_parser("nfc"); p.set_defaults(fn=cmd_nfc)
    p.add_argument("--language", default="hindi", choices=("hindi", "nepali"))
    p.add_argument("--shards", nargs="*", default=None)
    p.add_argument("--shard-limit", type=int, default=2)

    p = sub.add_parser("numerals"); p.set_defaults(fn=cmd_numerals)
    p.add_argument("--language", default="hindi", choices=("hindi", "nepali"))
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--shards", nargs="*", default=None)
    p.add_argument("--shard-limit", type=int, default=2)

    p = sub.add_parser("token-share"); p.set_defaults(fn=cmd_token_share)
    p.add_argument("--language", default="nepali", choices=("hindi", "nepali"))
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--manual-chars", type=int, required=True)
    p.add_argument("--auto-chars", type=int, required=True)
    p.add_argument("--manual-shards", nargs="+", required=True)
    p.add_argument("--auto-shards", nargs="+", required=True)
    p.add_argument("--sample-rate", type=float, default=0.04)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
