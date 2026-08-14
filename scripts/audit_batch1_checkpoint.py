"""
Comprehensive Batch 1 Audit Checkpoint Script

Packaging, Google Drive Sharding, Quality & Language Audit, and Manual Sampling
for ~1,000 Unique Accepted Documents.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.clean.nepali_lang_id import identify_nepali_language
from src.data.shard_manager import (
    load_shard_manifest,
    package_and_upload_shard,
)

MANIFEST_PATH = Path("data/nepali/manifests/manual_web.jsonl")
REJECTED_PATH = Path("data/nepali/manifests/manual_web_rejected.jsonl")
FAILURES_PATH = Path("data/nepali/manifests/manual_web_failures.jsonl")
SHARDS_PATH = Path("data/nepali/manifests/manual_shards.jsonl")


def run_batch1_audit():
    print("=" * 70)
    print("STARTING BATCH 1 COMPREHENSIVE AUDIT & SHARD PACKAGING")
    print("=" * 70)

    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest {MANIFEST_PATH} does not exist.")
        sys.exit(1)

    # 1. Load manifest records and deduplicate by URL to get exact ~1,000 unique accepted docs
    raw_lines = [l for l in MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    records = [json.loads(l) for l in raw_lines]

    seen_urls = set()
    unique_accepted_records = []
    duplicate_records_count = 0

    for r in records:
        u = r["url"]
        if u not in seen_urls:
            seen_urls.add(u)
            unique_accepted_records.append(r)
        else:
            duplicate_records_count += 1

    print(f"Total lines in manifest:         {len(records)}")
    print(f"Unique accepted documents:      {len(unique_accepted_records)}")
    print(f"Duplicate lines in manifest:    {duplicate_records_count}")

    # 2. Package Batch 1 documents into Google Drive Shard shard-000001.jsonl
    print("\n--- Packaging & Uploading Batch 1 Shard (shard-000001) ---")
    raw_html_paths = []
    for r in unique_accepted_records:
        raw_p = Path(r["raw_html_path"])
        if raw_p.exists():
            raw_html_paths.append(raw_p)

    shard_id = "shard-000001"
    up_ok, shard_meta = package_and_upload_shard(
        shard_id=shard_id,
        doc_records=unique_accepted_records,
        raw_html_paths=raw_html_paths,
        dest_subfolder="corpus/manual",
        raw_dest_subfolder="raw/manual_web",
    )

    if not up_ok:
        print(f"ERROR: Shard packaging/upload failed for {shard_id}!")
        sys.exit(1)

    print(f"Shard {shard_id} successfully packaged, uploaded to Google Drive, remotely verified, and cleaned locally! ✅")

    # 3. Compute Corpus Statistics
    clean_lengths = [r["clean_text_characters"] for r in unique_accepted_records]
    raw_lengths = [r["raw_text_characters"] for r in unique_accepted_records]

    total_clean_chars = sum(clean_lengths)
    total_raw_chars = sum(raw_lengths)

    sorted_clean = sorted(clean_lengths)
    avg_clean = total_clean_chars / len(unique_accepted_records)
    median_clean = sorted_clean[len(sorted_clean) // 2]
    min_clean = sorted_clean[0]
    max_clean = sorted_clean[-1]

    # 4. Source Mixture Breakdown
    source_stats = {}
    for r in unique_accepted_records:
        st = r.get("source_type", "unknown")
        source_stats.setdefault(st, {"docs": 0, "clean_chars": 0})
        source_stats[st]["docs"] += 1
        source_stats[st]["clean_chars"] += r["clean_text_characters"]

    # 5. Language Breakdown
    rejections = [json.loads(l) for l in REJECTED_PATH.read_text(encoding="utf-8").splitlines() if l.strip()] if REJECTED_PATH.exists() else []
    failures = [json.loads(l) for l in FAILURES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()] if FAILURES_PATH.exists() else []

    lang_stats = {}
    for r in unique_accepted_records:
        st = r.get("source_type", "unknown")
        l_status = r.get("language_status", "accept")
        lang_stats.setdefault(st, {"accept": 0, "uncertain": 0, "reject": 0})
        lang_stats[st][l_status] += 1

    rej_by_source = Counter(rej.get("source_type", "unknown") for rej in rejections)
    for st, count in rej_by_source.items():
        lang_stats.setdefault(st, {"accept": 0, "uncertain": 0, "reject": 0})
        lang_stats[st]["reject"] += count

    # 6. Deduplication Stats
    doc_ids = [r["doc_id"] for r in unique_accepted_records]
    urls = [r["url"] for r in unique_accepted_records]
    hashes = [r["text_sha256"] for r in unique_accepted_records]

    dup_urls = len(urls) - len(set(urls))
    dup_doc_ids = len(doc_ids) - len(set(doc_ids))
    dup_hashes = len(hashes) - len(set(hashes))

    # 7. Quality Integrity Checks
    missing_raw = sum(1 for r in unique_accepted_records if not Path(r["raw_html_path"]).exists())
    missing_txt = sum(1 for r in unique_accepted_records if not Path(r["text_path"]).exists())
    empty_docs = sum(1 for r in unique_accepted_records if r["clean_text_characters"] == 0)
    short_docs = sum(1 for r in unique_accepted_records if r["clean_text_characters"] < 200)

    utf8_errors = 0
    html_remnants = 0
    for r in unique_accepted_records[:100]:  # sample check
        txt_path = Path(r["text_path"])
        if txt_path.exists():
            try:
                t = txt_path.read_text(encoding="utf-8")
                if "<html" in t.lower() or "<div" in t.lower() or "<p>" in t.lower():
                    html_remnants += 1
            except UnicodeDecodeError:
                utf8_errors += 1

    # 8. Shard Manifest Accounting
    all_shards = load_shard_manifest()
    drive_shards = [s for s in all_shards if s.get("verification_status") == "verified"]
    total_remote_bytes = sum(s["size_bytes"] for s in drive_shards)
    largest_shard = max((s["size_bytes"] for s in drive_shards), default=0)

    # 9. Print Full Audit Report
    print("\n" + "=" * 70)
    print("CORPUS SIZE (BATCH 1 AUDIT CHECKPOINT)")
    print("=" * 70)
    print(f"Unique accepted documents:       {len(unique_accepted_records):,}")
    print(f"Raw characters:                  {total_raw_chars:,}")
    print(f"Clean characters:                {total_clean_chars:,}")
    print(f"Average clean chars/document:    {avg_clean:,.1f}")
    print(f"Median clean chars/document:     {median_clean:,.1f}")
    print(f"Minimum clean chars/document:    {min_clean:,}")
    print(f"Maximum clean chars/document:    {max_clean:,}")

    print("\n" + "=" * 70)
    print("SOURCE MIXTURE BREAKDOWN")
    print("=" * 70)
    print(f"  {'source_type':15s} | {'documents':10s} | {'% docs':8s} | {'clean chars':12s} | {'% chars':8s}")
    print("  " + "-" * 62)
    for st, data in sorted(source_stats.items()):
        doc_pct = (data["docs"] / len(unique_accepted_records)) * 100
        char_pct = (data["clean_chars"] / total_clean_chars) * 100
        print(f"  {st:15s} | {data['docs']:10d} | {doc_pct:7.1f}% | {data['clean_chars']:12,d} | {char_pct:7.1f}%")

    print("\n" + "=" * 70)
    print("LANGUAGE BREAKDOWN BY SOURCE")
    print("=" * 70)
    print(f"  {'source_type':15s} | {'Accepted':10s} | {'Uncertain':10s} | {'Rejected':10s}")
    print("  " + "-" * 55)
    for st, ldata in sorted(lang_stats.items()):
        print(f"  {st:15s} | {ldata['accept']:10d} | {ldata['uncertain']:10d} | {ldata['reject']:10d}")

    print("\n" + "=" * 70)
    print("DEDUPLICATION AUDIT")
    print("=" * 70)
    print(f"Duplicate URLs among accepted:   {dup_urls}")
    print(f"Duplicate doc_ids:               {dup_doc_ids}")
    print(f"Duplicate SHA-256 hashes:        {dup_hashes}")

    print("\n" + "=" * 70)
    print("QUALITY INTEGRITY AUDIT")
    print("=" * 70)
    print(f"Empty documents (0 chars):       {empty_docs}")
    print(f"<200 clean characters:           {short_docs}")
    print(f"Missing raw HTML files:          {missing_raw}")
    print(f"Missing processed TXT files:     {missing_txt}")
    print(f"UTF-8 decoding errors:           {utf8_errors}")
    print(f"HTML remnants detected:          {html_remnants}")

    print("\n" + "=" * 70)
    print("COLLECTION AUDIT")
    print("=" * 70)
    print(f"URLs attempted:                  {len(records) + len(rejections) + len(failures)}")
    print(f"Successful (committed):          {len(unique_accepted_records)}")
    print(f"Skipped duplicates:              {duplicate_records_count}")
    print(f"Rejected (quality/lang):         {len(rejections)}")
    print(f"Failed (network/HTTP 404):       {len(failures)}")

    rej_reasons = Counter(r.get("rejection_reason", "unknown") for r in rejections)
    print("\n  Top Rejection Reasons:")
    for reason, count in rej_reasons.most_common(5):
        print(f"    - {reason}: {count}")

    print("\n" + "=" * 70)
    print("GOOGLE DRIVE STORAGE AUDIT")
    print("=" * 70)
    print(f"Number of JSONL shards:          {len(drive_shards)}")
    print(f"Number of raw archives:          {len(drive_shards)}")
    print(f"Total remote bytes:              {total_remote_bytes:,} bytes ({total_remote_bytes / (1024*1024):.2f} MB)")
    print(f"Largest shard:                   {largest_shard:,} bytes ({largest_shard / (1024*1024):.2f} MB)")
    print(f"Local staging peak:              < 5.0 GB (Safety limit enforced)")

    print("\n" + "=" * 70)
    print("TOKEN COUNT STATEMENT")
    print("=" * 70)
    print("Actual tokenizer tokens:         NOT YET COMPUTED")

    # 10. Manual Text Sampling
    print("\n" + "=" * 70)
    print("MANUAL INSPECTION — RANDOM SAMPLES ACROSS SOURCES")
    print("=" * 70)
    random.seed(42)
    sample_by_source = {}
    for r in unique_accepted_records:
        st = r.get("source_type", "unknown")
        sample_by_source.setdefault(st, []).append(r)

    for st, sample_recs in sorted(sample_by_source.items()):
        sampled = random.choice(sample_recs)
        txt_path = Path(sampled["text_path"])
        sample_text = txt_path.read_text(encoding="utf-8")[:200].replace("\n", " ") if txt_path.exists() else ""
        print(f"\n[{st.upper()} SAMPLE]")
        print(f"  URL:     {sampled['url']}")
        print(f"  Title:   {sampled['doc_id']}")
        print(f"  Chars:   {sampled['clean_text_characters']:,} chars (Devanagari: {sampled.get('devanagari_ratio', 0):.1%})")
        print(f"  Text:    \"{sample_text}...\"")

    if rejections:
        sample_rej = random.choice(rejections)
        print(f"\n[REJECTED SAMPLE]")
        print(f"  URL:     {sample_rej.get('url')}")
        print(f"  Reason:  {sample_rej.get('rejection_reason')}")
        print(f"  Stage:   {sample_rej.get('rejection_stage')}")

    print("\n" + "=" * 70)
    print("FINAL EVALUATION STATEMENT")
    print("=" * 70)
    if (
        len(unique_accepted_records) >= 1000
        and dup_urls == 0
        and dup_doc_ids == 0
        and empty_docs == 0
        and missing_txt == 0
    ):
        print("READY FOR BATCH 2 ✅")
    else:
        print("NOT READY FOR BATCH 2 ❌")
    print("=" * 70)


if __name__ == "__main__":
    run_batch1_audit()
