"""
Corpus Accounting & Quality Validation Script for Nepali Manual Web Corpus

Reads data/nepali/manifests/manual_web.jsonl (and failures log if present)
and computes corpus statistics, source-type breakdowns, and quality audit checks.
"""

import json
from collections import Counter
from pathlib import Path

MANIFEST_PATH = Path("data/nepali/manifests/manual_web.jsonl")
FAILURES_PATH = Path("data/nepali/manifests/manual_web_failures.jsonl")
SOURCES_DIR = Path("data/nepali/sources")


def get_source_list_urls() -> set[str]:
    urls = set()
    if not SOURCES_DIR.exists():
        return urls

    for f in SOURCES_DIR.glob("*.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.add(line)
    return urls


def run_corpus_accounting():
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return

    manifest_lines = [
        line.strip()
        for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    records = [json.loads(line) for line in manifest_lines]
    num_docs = len(records)

    source_urls = get_source_list_urls()

    # Failures
    failures_count = 0
    if FAILURES_PATH.exists():
        failures_count = len(
            [
                line
                for line in FAILURES_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )

    if num_docs == 0:
        print("Manifest is empty.")
        return

    # Counts and metrics
    total_raw_chars = sum(r.get("raw_text_characters", 0) for r in records)
    total_clean_chars = sum(r.get("clean_text_characters", 0) for r in records)

    clean_lengths = [r.get("clean_text_characters", 0) for r in records]
    avg_clean_chars = total_clean_chars / num_docs if num_docs > 0 else 0
    min_clean_chars = min(clean_lengths) if clean_lengths else 0
    max_clean_chars = max(clean_lengths) if clean_lengths else 0

    # Source type breakdown
    docs_per_source = Counter(r.get("source_type", "unknown") for r in records)
    chars_per_source = Counter()
    for r in records:
        chars_per_source[r.get("source_type", "unknown")] += r.get(
            "clean_text_characters", 0
        )

    # Anomaly & Quality Audit Checks
    urls = [r["url"] for r in records]
    doc_ids = [r["doc_id"] for r in records]

    dup_urls = [url for url, count in Counter(urls).items() if count > 1]
    dup_ids = [did for did, count in Counter(doc_ids).items() if count > 1]

    empty_docs = [r for r in records if r.get("clean_text_characters", 0) == 0]
    short_docs = [
        r
        for r in records
        if 0 < r.get("clean_text_characters", 0) < 100
    ]

    missing_html_files = [
        r for r in records if not Path(r.get("raw_html_path", "")).exists()
    ]
    missing_txt_files = [
        r for r in records if not Path(r.get("text_path", "")).exists()
    ]

    utf8_errors = []
    for r in records:
        txt_path = Path(r.get("text_path", ""))
        if txt_path.exists():
            try:
                txt_path.read_text(encoding="utf-8")
            except Exception as e:
                utf8_errors.append((txt_path, str(e)))

    # Print Report
    print("=" * 60)
    print("        NEPALI MANUAL WEB CORPUS STATISTICS REPORT")
    print("=" * 60)
    print(f"Total Documents:              {num_docs:,}")
    print(f"Total Raw Characters:         {total_raw_chars:,}")
    print(f"Total Clean Characters:       {total_clean_chars:,}")
    print(f"Average Clean Chars/Doc:      {avg_clean_chars:,.1f}")
    print(f"Min Document Length (chars):  {min_clean_chars:,}")
    print(f"Max Document Length (chars):  {max_clean_chars:,}")
    print(f"Source List URLs Total:       {len(source_urls):,}")
    print(f"URLs Collected / Skipped:     {len(records):,} / {len(source_urls & set(urls)):,}")
    print(f"Total Logged Failures:        {failures_count:,}")
    print("-" * 60)
    print("BREAKDOWN BY SOURCE TYPE:")
    for stype in sorted(docs_per_source.keys()):
        d_cnt = docs_per_source[stype]
        c_cnt = chars_per_source[stype]
        print(f"  [{stype}] {d_cnt:,} docs | {c_cnt:,} clean chars")
    print("-" * 60)
    print("QUALITY AUDIT CHECKS:")
    print(f"  Duplicate URLs:             {len(dup_urls)}")
    print(f"  Duplicate doc_ids:          {len(dup_ids)}")
    print(f"  Empty Documents (0 chars):  {len(empty_docs)}")
    print(f"  Short Documents (<100 chars): {len(short_docs)}")
    print(f"  Missing HTML files:         {len(missing_html_files)}")
    print(f"  Missing TXT files:          {len(missing_txt_files)}")
    print(f"  UTF-8 Decoding Errors:      {len(utf8_errors)}")
    print("=" * 60)

    all_valid = (
        len(dup_urls) == 0
        and len(dup_ids) == 0
        and len(empty_docs) == 0
        and len(missing_html_files) == 0
        and len(missing_txt_files) == 0
        and len(utf8_errors) == 0
    )

    if all_valid:
        print("Status: ALL QUALITY CHECKS PASSED SUCCESSFULLY ✅")
    else:
        print("Status: AUDIT WARNING - Issues detected! ❌")
    print("=" * 60)


if __name__ == "__main__":
    run_corpus_accounting()
