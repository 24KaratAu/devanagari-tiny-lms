"""Report exact Hindi corpus accounting straight from Google Drive.

Streams the cleaned shards with `rclone cat` (local disk here has < 4 GB free,
so nothing is downloaded to stay) and reports, per source group, the document
count, character count and real BPE token count using the project's own
tokenizer. Token counts come from a sampled tokenisation extrapolated by
characters, which keeps a full pass over ~7 GB of JSONL affordable.

The output is the table the Phase 1 report needs, plus the two numbers that
decide whether the corpus is admissible: total tokens and the manual share.

Usage:
    python scripts/audit_hindi_corpus.py --manual-only
    python scripts/audit_hindi_corpus.py --auto-sample 8 --out report/hindi_audit.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_REMOTE = "gdrive:24KaratAu/hindi/cleaned"
MIN_MANUAL_SHARE = 0.20
TOTAL_TOKEN_TARGET = 500_000_000


def source_group(name: str) -> str:
    if name.startswith("wiki_shard"):
        return "Manual — Wikipedia dump"
    if name.startswith(("wikisource", "manual_wikisource")):
        return "Manual — Wikisource dump"
    if name.startswith("manual_lit"):
        return "Manual — Literature dump"
    if name.startswith("manual_aux"):
        return "Manual — Wikibooks / Wikiquote"
    if name.startswith("manual_news"):
        return "Manual — News crawl (Colab)"
    if name.startswith("manual_web"):
        return "Manual — Web crawl (sitemap)"
    if name.startswith("manual_"):
        return "Manual — Other"
    return "Automated — Sangraha Verified"


def list_shards(remote: str, sub: str) -> list[dict]:
    proc = subprocess.run(["rclone", "lsjson", f"{remote}/{sub}"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  (no {sub}: {proc.stderr.strip()[:120]})", file=sys.stderr)
        return []
    return sorted((e for e in json.loads(proc.stdout) if e["Name"].endswith(".jsonl")),
                  key=lambda e: e["Name"])


def _stream_shard(remote: str, sub: str, name: str, expected_size: int,
                  attempts: int = 4):
    """Stream one shard, verifying the byte count against Drive's file size.

    A truncated `rclone cat` through a pipe is indistinguishable from a short
    file: the reader just sees EOF early and reports a plausible-looking
    number. An earlier version of this script lost exactly two 5,000-document
    shards that way and under-reported the corpus by 10,000 documents with no
    error anywhere. Anything that reads a remote shard has to check that it got
    all the bytes.
    """
    for attempt in range(attempts):
        proc = subprocess.Popen(["rclone", "cat", f"{remote}/{sub}/{name}",
                                 "--retries", "5", "--timeout", "60s"],
                                stdout=subprocess.PIPE)
        docs, read_bytes = [], 0
        for raw in proc.stdout:
            read_bytes += len(raw)
            if raw.strip():
                docs.append(raw)
        proc.stdout.close()
        proc.wait()
        if read_bytes == expected_size:
            return docs
        print(f"    {name}: short read {read_bytes:,} != {expected_size:,} "
              f"(attempt {attempt + 1}/{attempts})", file=sys.stderr, flush=True)
    raise RuntimeError(
        f"{name}: could not read all {expected_size:,} bytes after {attempts} "
        f"attempts; refusing to report a partial count")


def scan(remote: str, sub: str, shards: list[dict], sp, sample_rate: float) -> dict:
    import random
    rng = random.Random(42)
    groups: dict[str, dict] = defaultdict(
        lambda: {"shards": 0, "docs": 0, "chars": 0, "s_chars": 0, "s_tokens": 0})
    for entry in shards:
        name = entry["Name"]
        g = groups[source_group(name)]
        g["shards"] += 1
        for raw in _stream_shard(remote, sub, name, entry["Size"]):
            try:
                doc = json.loads(raw)
            except Exception:
                continue
            text = doc.get("text", "")
            if not text.strip():
                continue
            g["docs"] += 1
            g["chars"] += len(text)
            if sp is not None and rng.random() < sample_rate:
                g["s_chars"] += len(text)
                g["s_tokens"] += len(sp.encode(text))
        print(f"    {name}: {g['docs']:,} docs / {g['chars']:,} chars",
              file=sys.stderr, flush=True)
    return groups


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--remote", default=DEFAULT_REMOTE)
    ap.add_argument("--tokenizer", default=None,
                    help="Path to hindi_bpe_8k.model (downloaded from Drive if omitted).")
    ap.add_argument("--sample-rate", type=float, default=0.05)
    ap.add_argument("--auto-sample", type=int, default=8,
                    help="Automated shards to tokenise; totals extrapolate by byte size.")
    ap.add_argument("--manual-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sp = None
    model = args.tokenizer
    if model is None:
        model = "/tmp/hindi_bpe_8k.model"
        subprocess.run(["rclone", "copyto", "gdrive:24KaratAu/hindi/tokenizer/hindi_bpe_8k.model",
                        model], capture_output=True)
    if Path(model).exists():
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(model)
    else:
        print("No tokenizer available; reporting characters only.", file=sys.stderr)

    manual = list_shards(args.remote, "manual")
    print(f"manual: {len(manual)} shards", file=sys.stderr)
    groups = scan(args.remote, "manual", manual, sp, args.sample_rate)

    auto_total_bytes = 0
    if not args.manual_only:
        auto = list_shards(args.remote, "automated_nonweb")
        auto_total_bytes = sum(e["Size"] for e in auto)
        step = max(len(auto) // max(args.auto_sample, 1), 1)
        picked = auto[::step][:args.auto_sample]
        print(f"automated: {len(auto)} shards, tokenising {len(picked)}", file=sys.stderr)
        ag = scan(args.remote, "automated_nonweb", picked, sp, args.sample_rate)
        picked_bytes = sum(e["Size"] for e in picked)
        scale = auto_total_bytes / picked_bytes if picked_bytes else 1.0
        for key, g in ag.items():
            g["shards"] = len(auto)
            g["docs"] = int(g["docs"] * scale)
            g["chars"] = int(g["chars"] * scale)
            g["extrapolated_from_shards"] = len(picked)
            groups[key] = g

    rows, manual_tokens, auto_tokens = [], 0.0, 0.0
    for name, g in groups.items():
        cpt = (g["s_chars"] / g["s_tokens"]) if g["s_tokens"] else None
        tokens = int(g["chars"] / cpt) if cpt else None
        rows.append((name, g["shards"], g["docs"], g["chars"], cpt, tokens))
        if tokens:
            if name.startswith("Manual"):
                manual_tokens += tokens
            else:
                auto_tokens += tokens
    rows.sort(key=lambda r: (not r[0].startswith("Manual"), -r[3]))

    print()
    print(f'{"Source":<34}{"Shards":>8}{"Docs":>12}{"Chars":>17}{"c/tok":>8}{"Tokens":>15}')
    print("-" * 94)
    for name, shards, docs, chars, cpt, tokens in rows:
        print(f'{name:<34}{shards:>8}{docs:>12,}{chars:>17,}'
              f'{(f"{cpt:.3f}" if cpt else "-"):>8}{(f"{tokens:,}" if tokens else "-"):>15}')
    print("-" * 94)

    if manual_tokens:
        pool_total = manual_tokens + auto_tokens
        print(f"Manual pool     : {int(manual_tokens):>15,} tokens")
        if auto_tokens:
            print(f"Automated pool  : {int(auto_tokens):>15,} tokens")
            print(f"Pool total      : {int(pool_total):>15,} tokens "
                  f"(manual {manual_tokens / pool_total * 100:.1f}%)")
        # The pool is bigger than the trainable corpus: the 20% floor means the
        # usable total is capped at 5x manual, however much automated data exists.
        reachable = manual_tokens / MIN_MANUAL_SHARE
        print()
        print(f"Max corpus at the {MIN_MANUAL_SHARE:.0%} manual floor: "
              f"{int(reachable):,} tokens")
        if reachable >= TOTAL_TOKEN_TARGET:
            share = manual_tokens / TOTAL_TOKEN_TARGET * 100
            print(f"OK  {TOTAL_TOKEN_TARGET:,}-token target reachable "
                  f"(manual would be {share:.1f}%)")
        else:
            need_chars_hint = ""
            deficit = TOTAL_TOKEN_TARGET * MIN_MANUAL_SHARE - manual_tokens
            print(f"FAIL Short by {int(deficit):,} manual tokens for a "
                  f"{TOTAL_TOKEN_TARGET:,}-token corpus{need_chars_hint}")

    if args.out:
        payload = {"rows": [dict(zip(
            ("source", "shards", "docs", "chars", "chars_per_token", "tokens"), r))
            for r in rows],
            "manual_tokens": int(manual_tokens),
            "automated_tokens": int(auto_tokens)}
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
