"""Process a Hindi Wikimedia XML dump into cleaned manual shards on Google Drive.

Used for the auxiliary Hindi projects that the Colab run either skipped or
produced almost nothing from (Wikisource returned a single document). Runs
locally, streams the bz2 dump with iterparse, and uploads each shard with
rclone so local disk never holds more than one shard.

Usage:
    python scripts/process_hindi_wikimedia_dump.py \
        --dump /path/hiwikisource.xml.bz2 --project wikisource
    # add --diagnose to print filter-rejection counts without writing anything
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.clean.clean_hindi_web import (  # noqa: E402
    devanagari_ratio, latin_ratio,
)

DRIVE_MANUAL_DIR = "gdrive:24KaratAu/hindi/cleaned/manual"

# Machine-translation maintenance banners: the course explicitly rules out
# machine-translated text, so any page carrying one of these is dropped.
MT_TAGS_PATTERN = re.compile(
    r"\{\{\s*(?:अनुवाद|अशुद्ध\s*अनुवाद|मशीनी\s*अनुवाद|यांत्रिक\s*अनुवाद|अनुवाद\s*अधूरा|"
    r"rough\s*translation|bad\s*translation|cleanup-translation|machine\s*translation|"
    r"not\s*hindi|translate|translateme)",
    re.IGNORECASE,
)

WIKI_MARKUP = [
    (re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL), ""),
    (re.compile(r"<ref[^/>]*/>"), ""),
    (re.compile(r"<!--.*?-->", re.DOTALL), ""),
    (re.compile(r"\{\{[^{}]*\}\}"), ""),          # innermost templates
    (re.compile(r"\[\[(?:Category|श्रेणी|वर्ग|File|चित्र|Image):[^\]]*\]\]"), ""),
    (re.compile(r"\[\[[^|\]]*\|([^\]]*)\]\]"), r"\1"),
    (re.compile(r"\[\[([^\]]*)\]\]"), r"\1"),
    (re.compile(r"\[https?://[^\s\]]*\s*([^\]]*)\]"), r"\1"),
    (re.compile(r"\[https?://[^\]]*\]"), ""),
    (re.compile(r"'{2,5}"), ""),
    (re.compile(r"<[^>]+>"), " "),
    (re.compile(r"&[a-zA-Z]+;|&#\d+;"), " "),
    (re.compile(r"\[\d{1,3}\]"), ""),
    (re.compile(r"^[=]{2,}\s*(.*?)\s*[=]{2,}$", re.MULTILINE), r"\1"),
    (re.compile(r"^[*#:;]+\s*", re.MULTILINE), ""),
]

TABLE_PAT = [
    (re.compile(r"^\s*\{\|.*$", re.MULTILINE), ""),
    (re.compile(r"^\s*\|\}.*$", re.MULTILINE), ""),
    (re.compile(r"^\s*\|[-+].*$", re.MULTILINE), ""),
    (re.compile(r"^\s*!.*$", re.MULTILINE), ""),
]

SKIP_SEC = {
    "सन्दर्भ", "संदर्भ", "बाहरी कड़ियाँ", "बाहरी कड़ी", "श्रेणी",
    "इन्हें भी देखें", "यह भी देखें", "स्रोत", "टिप्पणी", "नोट",
    "references", "external links", "see also", "notes",
}


def strip_templates(text: str, rounds: int = 6) -> str:
    """Remove nested {{...}} by repeatedly deleting the innermost pairs."""
    inner = re.compile(r"\{\{[^{}]*\}\}")
    for _ in range(rounds):
        text, n = inner.subn("", text)
        if not n:
            break
    return text


def clean_wiki(text: str) -> str:
    for pat, rep in TABLE_PAT:
        text = pat.sub(rep, text)
    text = strip_templates(text)
    for pat, rep in WIKI_MARKUP:
        text = pat.sub(rep, text)
    kept, skip = [], False
    for line in text.split("\n"):
        stripped = line.strip().lower()
        if stripped in SKIP_SEC:
            skip = True
            continue
        if re.match(r"^={2,}.*={2,}$", stripped):
            skip = False
        if not skip:
            kept.append(line)
    text = "\n".join(kept)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_pages(dump_path: Path):
    """Yield (namespace, title, page_id, wikitext) with flat memory usage."""
    with bz2.open(dump_path, "rt", encoding="utf-8") as fh:
        context = iter(ET.iterparse(fh, events=("start", "end")))
        _, root = next(context)
        for event, elem in context:
            tag = elem.tag.rsplit("}", 1)[-1]
            if event != "end" or tag != "page":
                continue
            get = lambda parent, name: next(  # noqa: E731
                (c for c in parent if c.tag.rsplit("}", 1)[-1] == name), None)
            ns_el = get(elem, "ns")
            title_el = get(elem, "title")
            id_el = get(elem, "id")
            redirect_el = get(elem, "redirect")
            rev = get(elem, "revision")
            text = ""
            if rev is not None:
                te = get(rev, "text")
                if te is not None and te.text:
                    text = te.text
            ns = ns_el.text if ns_el is not None else None
            yield (ns,
                   title_el.text if title_el is not None else "",
                   id_el.text if id_el is not None else "",
                   text,
                   redirect_el is not None)
            elem.clear()
            root.clear()


def upload(path: Path, remote: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry-run] kept {path.name} locally")
        return
    proc = subprocess.run(
        ["rclone", "moveto", str(path), f"{remote}/{path.name}",
         "--retries", "5", "--low-level-retries", "10"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  ⚠️  upload failed for {path.name}: {proc.stderr.strip()[:300]}",
              file=sys.stderr)
    else:
        print(f"  ⬆️  {path.name} → Drive")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--project", required=True,
                    help="Short name used in doc metadata and shard filenames.")
    ap.add_argument("--namespaces", default="0,104",
                    help="Comma-separated dump namespaces to keep.")
    ap.add_argument("--remote", default=DRIVE_MANUAL_DIR)
    ap.add_argument("--staging", default=None)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--min-dev-ratio", type=float, default=0.50)
    ap.add_argument("--max-latin-ratio", type=float, default=0.10)
    ap.add_argument("--diagnose", type=int, default=0,
                    help="Inspect the first N pages, report rejection reasons, write nothing.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    keep_ns = {n.strip() for n in args.namespaces.split(",") if n.strip()}
    staging = Path(args.staging) if args.staging else Path(args.dump).parent / "staging"
    staging.mkdir(parents=True, exist_ok=True)

    reasons = Counter()
    ns_counter = Counter()
    buffer: list[dict] = []
    shard_index = args.start_index
    total_docs = total_chars = 0
    seen: set[str] = set()

    for i, (ns, title, pid, raw, is_redirect) in enumerate(iter_pages(Path(args.dump))):
        if args.diagnose and i >= args.diagnose:
            break
        ns_counter[ns] += 1
        if is_redirect:
            reasons["redirect"] += 1
            continue
        if ns not in keep_ns:
            reasons[f"ns_{ns}"] += 1
            continue
        if not raw or not raw.strip():
            reasons["empty"] += 1
            continue
        if MT_TAGS_PATTERN.search(raw):
            reasons["machine_translation_banner"] += 1
            continue
        cleaned = clean_wiki(raw)
        if len(cleaned) < args.min_chars:
            reasons["too_short"] += 1
            continue
        dev = devanagari_ratio(cleaned)
        lat = latin_ratio(cleaned)
        if dev < args.min_dev_ratio:
            reasons["low_devanagari"] += 1
            continue
        if lat > args.max_latin_ratio:
            reasons["high_latin"] += 1
            continue
        sha = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        if sha in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(sha)
        reasons["accepted"] += 1
        total_docs += 1
        total_chars += len(cleaned)
        if args.diagnose:
            continue
        buffer.append({
            "doc_id": f"{args.project}_{pid}",
            "title": title,
            "wiki_page_id": pid,
            "wiki_namespace": ns,
            "source": f"Hindi {args.project}",
            "source_type": f"{args.project}_dump",
            "collection_type": "manual",
            "language": "hindi",
            "text": cleaned,
            "clean_text_characters": len(cleaned),
            "devanagari_ratio": round(dev, 4),
            "latin_ratio": round(lat, 4),
            "text_sha256": sha,
        })
        if len(buffer) >= args.shard_size:
            path = staging / f"manual_{args.project}_shard_{shard_index:04d}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for doc in buffer:
                    fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
            print(f"  wrote {path.name}: {len(buffer):,} docs "
                  f"({total_chars:,} chars cumulative)", flush=True)
            upload(path, args.remote, args.dry_run)
            buffer = []
            shard_index += 1

    if buffer:
        path = staging / f"manual_{args.project}_shard_{shard_index:04d}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for doc in buffer:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"  wrote {path.name}: {len(buffer):,} docs", flush=True)
        upload(path, args.remote, args.dry_run)
        shard_index += 1

    print("=" * 60)
    print(f"{args.project}: {total_docs:,} docs / {total_chars:,} clean chars "
          f"(~{int(total_chars / 3.19):,} tokens)")
    print("Namespace histogram:", dict(ns_counter.most_common(10)))
    print("Outcomes:")
    for reason, n in reasons.most_common():
        print(f"  {reason:32s} {n:,}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
