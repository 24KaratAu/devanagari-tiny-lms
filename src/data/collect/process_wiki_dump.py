"""
Wikipedia Dump Processor

Downloads and processes Wikipedia XML dumps for Nepali and Hindi.
Extracts article text, cleans it, and outputs JSONL shards ready for
tokenizer training.

This counts as "manual collection" per TA clarification:
  "Wikipedia dumps are considered manual collection, provided you fetch,
   process, clean, and organize the raw data yourself."

Usage:
    python -m src.data.collect.process_wiki_dump \
        --language nepali \
        --dump-path data/nepali/raw/wiki_dump/newiki-latest-pages-articles.xml.bz2 \
        --output-dir data/nepali/processed/wiki_dump/ \
        --shard-size 10000
"""

import argparse
import bz2
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone
from typing import Generator


# ── Devanagari script detection ────────────────────────────────────────────

def devanagari_ratio(text: str) -> float:
    """Calculate the ratio of Devanagari characters in text."""
    if not text:
        return 0.0
    dev_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    return dev_chars / len(text)


def latin_ratio(text: str) -> float:
    """Calculate the ratio of Latin characters in text."""
    if not text:
        return 0.0
    lat_chars = sum(1 for c in text if "a" <= c.lower() <= "z")
    return lat_chars / len(text)


# ── Text cleaning ─────────────────────────────────────────────────────────

# MediaWiki markup patterns to strip
WIKI_MARKUP_PATTERNS = [
    (re.compile(r"\{\{[^}]*\}\}"), ""),           # Templates {{...}}
    (re.compile(r"\[\[Category:[^\]]*\]\]"), ""),  # Categories
    (re.compile(r"\[\[श्रेणी:[^\]]*\]\]"), ""),    # Nepali categories
    (re.compile(r"\[\[वर्ग:[^\]]*\]\]"), ""),      # Hindi categories
    (re.compile(r"\[\[[^|\]]*\|([^\]]*)\]\]"), r"\1"),  # [[link|text]] → text
    (re.compile(r"\[\[([^\]]*)\]\]"), r"\1"),       # [[link]] → link
    (re.compile(r"\[https?://[^\s\]]*\s*([^\]]*)\]"), r"\1"),  # External links
    (re.compile(r"\[https?://[^\]]*\]"), ""),        # Bare external links
    (re.compile(r"'{2,5}"), ""),                     # Bold/italic markers
    (re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL), ""),  # <ref>...</ref>
    (re.compile(r"<ref[^/>]*/>"), ""),               # <ref ... />
    (re.compile(r"<[^>]+>"), ""),                     # Any remaining HTML tags
    (re.compile(r"&[a-zA-Z]+;"), " "),               # HTML entities
    (re.compile(r"&[#]\d+;"), " "),                  # Numeric HTML entities
    (re.compile(r"\[\d+\]"), ""),                     # Citation markers [1]
]

# Table / list markup
TABLE_LIST_PATTERNS = [
    (re.compile(r"^\s*\{\|.*$", re.MULTILINE), ""),  # Table start
    (re.compile(r"^\s*\|\}.*$", re.MULTILINE), ""),  # Table end
    (re.compile(r"^\s*\|-.*$", re.MULTILINE), ""),   # Table row separator
    (re.compile(r"^\s*!.*$", re.MULTILINE), ""),     # Table header
    (re.compile(r"^\|.*$", re.MULTILINE), ""),       # Table cell (aggressive)
]

# Section headers
HEADER_PATTERN = re.compile(r"^={2,6}\s*(.*?)\s*={2,6}\s*$", re.MULTILINE)

# Sections to skip entirely (common non-content sections)
SKIP_SECTIONS = {
    # Nepali
    "सन्दर्भ", "सन्दर्भहरू", "बाह्य कडीहरू", "बाह्य लिङ्कहरू",
    "श्रेणी", "यो पनि हेर्नुहोस्", "स्रोत", "टिप्पणी", "टिप्पणीहरू",
    # Hindi
    "सन्दर्भ", "संदर्भ", "बाहरी कड़ियाँ", "बाहरी कड़ी", "श्रेणी",
    "इन्हें भी देखें", "यह भी देखें", "स्रोत", "टिप्पणी", "नोट",
    # English (sometimes mixed in)
    "references", "external links", "see also", "notes", "further reading",
    "bibliography", "sources",
}


def clean_wiki_markup(text: str) -> str:
    """Remove MediaWiki markup from raw article text."""
    # Remove table markup first
    for pattern, repl in TABLE_LIST_PATTERNS:
        text = pattern.sub(repl, text)

    # Remove wiki markup patterns
    for pattern, repl in WIKI_MARKUP_PATTERNS:
        text = pattern.sub(repl, text)

    # Convert headers to plain text with newlines
    text = HEADER_PATTERN.sub(r"\n\1\n", text)

    return text


def remove_non_content_sections(text: str) -> str:
    """Remove reference/external-link sections and everything after them."""
    lines = text.split("\n")
    result = []
    skip_mode = False

    for line in lines:
        stripped = line.strip()

        # Check if this is a section header
        header_match = re.match(r"^(.*?)$", stripped)
        if header_match:
            section_title = header_match.group(1).strip().lower()
            if section_title in SKIP_SECTIONS:
                skip_mode = True
                continue
            else:
                skip_mode = False

        if not skip_mode:
            result.append(line)

    return "\n".join(result)


def normalize_text(text: str) -> str:
    """Normalize whitespace and remove junk lines."""
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        # Skip very short lines that are likely artifacts
        if len(line) < 3 and not line:
            lines.append("")
            continue
        # Skip lines that are mostly punctuation/special chars
        if line and len(line) > 0:
            alnum_ratio = sum(1 for c in line if c.isalnum() or "\u0900" <= c <= "\u097F") / len(line)
            if alnum_ratio < 0.3 and len(line) < 20:
                continue
        lines.append(line)

    # Collapse multiple blank lines
    result = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                result.append("")
            prev_blank = True
        else:
            prev_blank = False
            result.append(line)

    return "\n".join(result).strip()


MT_TAGS_PATTERN = re.compile(
    r"\{\{\s*(?:अनुवाद|अशुद्ध\s*अनुवाद|मशीनी\s*अनुवाद|यांत्रिक\s*अनुवाद|अनुवाद\s*अधूरा|"
    r"rough\s*translation|bad\s*translation|cleanup-translation|machine\s*translation|"
    r"not\s*hindi|translate|translateme)[^}]*\}\}",
    re.IGNORECASE,
)


def clean_article(text: str) -> str:
    """Full cleaning pipeline for a single Wikipedia article. Returns empty if MT-flagged."""
    if MT_TAGS_PATTERN.search(text):
        return ""
    text = clean_wiki_markup(text)
    text = remove_non_content_sections(text)
    text = normalize_text(text)
    return text


# ── XML Dump Parser ───────────────────────────────────────────────────────

NAMESPACE = "{http://www.mediawiki.org/xml/export-0.10/}"


def iter_wiki_articles(dump_path: str) -> Generator[dict, None, None]:
    """Stream-parse a Wikipedia XML dump and yield articles.

    Yields dicts with keys: title, text, id, ns (namespace).
    Only yields namespace 0 (main articles), skipping redirects.
    """
    # Handle both .xml.bz2 and plain .xml
    if dump_path.endswith(".bz2"):
        file_obj = bz2.open(dump_path, "rt", encoding="utf-8")
    else:
        file_obj = open(dump_path, "r", encoding="utf-8")

    try:
        context = ET.iterparse(file_obj, events=("end",))
        for event, elem in context:
            if elem.tag == f"{NAMESPACE}page":
                ns_elem = elem.find(f"{NAMESPACE}ns")
                ns = ns_elem.text if ns_elem is not None else "0"

                # Only main namespace articles
                if ns != "0":
                    elem.clear()
                    continue

                # Skip redirects
                redirect = elem.find(f"{NAMESPACE}redirect")
                if redirect is not None:
                    elem.clear()
                    continue

                title_elem = elem.find(f"{NAMESPACE}title")
                id_elem = elem.find(f"{NAMESPACE}id")
                revision = elem.find(f"{NAMESPACE}revision")

                title = title_elem.text if title_elem is not None else ""
                page_id = id_elem.text if id_elem is not None else ""

                text_content = ""
                if revision is not None:
                    text_elem = revision.find(f"{NAMESPACE}text")
                    if text_elem is not None and text_elem.text:
                        text_content = text_elem.text

                if text_content.strip():
                    yield {
                        "title": title,
                        "text": text_content,
                        "id": page_id,
                    }

                elem.clear()
    finally:
        file_obj.close()


# ── Main Processing Pipeline ──────────────────────────────────────────────

def process_wiki_dump(
    dump_path: str,
    output_dir: str,
    language: str = "nepali",
    shard_size: int = 10000,
    min_clean_chars: int = 100,
    min_devanagari_ratio: float = 0.5,
):
    """Process a Wikipedia XML dump into clean JSONL shards.

    Args:
        dump_path: Path to the .xml.bz2 dump file.
        output_dir: Directory to write JSONL shard files.
        language: "nepali" or "hindi".
        shard_size: Number of documents per shard file.
        min_clean_chars: Minimum cleaned character count to keep an article.
        min_devanagari_ratio: Minimum Devanagari script ratio to keep.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    doc_count = 0
    shard_docs = []
    total_accepted = 0
    total_rejected = 0
    total_chars = 0
    total_articles_seen = 0

    print(f"Processing {language} Wikipedia dump: {dump_path}")
    print(f"Output directory: {output_dir}")
    print(f"Shard size: {shard_size} docs/shard")
    print(f"Min clean chars: {min_clean_chars}, Min Devanagari ratio: {min_devanagari_ratio}")
    print("=" * 60)

    for article in iter_wiki_articles(dump_path):
        total_articles_seen += 1

        if total_articles_seen % 5000 == 0:
            print(
                f"  Processed {total_articles_seen:,} articles | "
                f"Accepted: {total_accepted:,} | Rejected: {total_rejected:,} | "
                f"Total chars: {total_chars:,}"
            )

        # Clean the article
        cleaned = clean_article(article["text"])

        # Quality checks
        if len(cleaned) < min_clean_chars:
            total_rejected += 1
            continue

        dev_r = devanagari_ratio(cleaned)
        if dev_r < min_devanagari_ratio:
            total_rejected += 1
            continue

        # Build record
        text_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        doc_id = hashlib.md5(
            f'{article["id"]}_{article["title"]}'.encode("utf-8")
        ).hexdigest()[:16]

        record = {
            "doc_id": doc_id,
            "title": article["title"],
            "wiki_page_id": article["id"],
            "source": f"{language.capitalize()} Wikipedia Dump",
            "source_type": "wikipedia_dump",
            "collection_type": "manual",
            "language": language,
            "text": cleaned,
            "clean_text_characters": len(cleaned),
            "devanagari_ratio": round(dev_r, 4),
            "latin_ratio": round(latin_ratio(cleaned), 4),
            "text_sha256": text_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        shard_docs.append(record)
        total_accepted += 1
        total_chars += len(cleaned)

        # Write shard when full
        if len(shard_docs) >= shard_size:
            shard_file = output_path / f"wiki_shard_{shard_idx:04d}.jsonl"
            with open(shard_file, "w", encoding="utf-8") as f:
                for doc in shard_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            print(
                f"  OK  Wrote {shard_file.name}: {len(shard_docs):,} docs, "
                f"{sum(d['clean_text_characters'] for d in shard_docs):,} chars"
            )
            shard_docs = []
            shard_idx += 1

    # Write remaining docs
    if shard_docs:
        shard_file = output_path / f"wiki_shard_{shard_idx:04d}.jsonl"
        with open(shard_file, "w", encoding="utf-8") as f:
            for doc in shard_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(
            f"  OK  Wrote {shard_file.name}: {len(shard_docs):,} docs, "
            f"{sum(d['clean_text_characters'] for d in shard_docs):,} chars"
        )

    print("\n" + "=" * 60)
    print(f"WIKI DUMP PROCESSING COMPLETE — {language.upper()}")
    print(f"  Total articles seen:     {total_articles_seen:,}")
    print(f"  Total accepted:          {total_accepted:,}")
    print(f"  Total rejected:          {total_rejected:,}")
    print(f"  Total clean characters:  {total_chars:,}")
    print(f"  Shards written:          {shard_idx + 1}")
    print("=" * 60)

    # Write summary manifest
    summary = {
        "language": language,
        "dump_path": str(dump_path),
        "total_articles_seen": total_articles_seen,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "total_clean_characters": total_chars,
        "shards_written": shard_idx + 1,
        "shard_size": shard_size,
        "min_clean_chars": min_clean_chars,
        "min_devanagari_ratio": min_devanagari_ratio,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = output_path / "processing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Process Wikipedia XML dump into clean JSONL shards")
    parser.add_argument("--language", required=True, choices=["nepali", "hindi"],
                        help="Target language")
    parser.add_argument("--dump-path", required=True,
                        help="Path to Wikipedia XML dump (.xml.bz2)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output JSONL shards")
    parser.add_argument("--shard-size", type=int, default=10000,
                        help="Documents per shard (default: 10000)")
    parser.add_argument("--min-chars", type=int, default=100,
                        help="Minimum cleaned characters to keep (default: 100)")
    parser.add_argument("--min-devanagari-ratio", type=float, default=0.5,
                        help="Minimum Devanagari character ratio (default: 0.5)")

    args = parser.parse_args()

    process_wiki_dump(
        dump_path=args.dump_path,
        output_dir=args.output_dir,
        language=args.language,
        shard_size=args.shard_size,
        min_clean_chars=args.min_chars,
        min_devanagari_ratio=args.min_devanagari_ratio,
    )


if __name__ == "__main__":
    main()
