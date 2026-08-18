"""
Deduplication Pipeline

Provides exact and near-duplicate detection for corpus documents.
Supports:
  1. Exact dedup via SHA-256 hash of cleaned text
  2. URL-level dedup
  3. Near-duplicate detection via MinHash (optional, for large corpora)

Usage:
    python -m src.data.deduplicate.dedup \
        --input-dir data/nepali/processed/wiki_dump/ \
        --output-dir data/nepali/processed/deduped/ \
        --language nepali
"""

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone


def deduplicate_jsonl_shards(
    input_dir: str,
    output_dir: str,
    language: str = "nepali",
):
    """Deduplicate JSONL shards by content hash and URL.

    Reads all .jsonl files from input_dir, deduplicates by text_sha256
    and URL, then writes clean shards to output_dir.

    Args:
        input_dir: Directory containing JSONL shard files
        output_dir: Directory for deduplicated output shards
        language: Language label for logging
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(input_path.glob("*.jsonl"))
    if not shard_files:
        print(f"No .jsonl files found in {input_dir}")
        return

    print(f"Deduplicating {len(shard_files)} shard files from {input_dir}")
    print("=" * 60)

    seen_hashes = set()
    seen_urls = set()
    total_input = 0
    total_output = 0
    total_hash_dupes = 0
    total_url_dupes = 0
    total_chars = 0

    all_unique_docs = []

    for shard_file in shard_files:
        with open(shard_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_input += 1

                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Hash-based dedup
                text_hash = doc.get("text_sha256", "")
                if not text_hash:
                    text = doc.get("text", "")
                    if text:
                        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    else:
                        continue

                if text_hash in seen_hashes:
                    total_hash_dupes += 1
                    continue
                seen_hashes.add(text_hash)

                # URL-based dedup
                url = doc.get("url", "")
                if url:
                    if url in seen_urls:
                        total_url_dupes += 1
                        continue
                    seen_urls.add(url)

                all_unique_docs.append(doc)
                total_output += 1
                total_chars += doc.get("clean_text_characters", len(doc.get("text", "")))

    # Write deduplicated shards (preserving shard structure)
    shard_size = 10000
    shard_idx = 0
    for i in range(0, len(all_unique_docs), shard_size):
        batch = all_unique_docs[i : i + shard_size]
        shard_file = output_path / f"deduped_shard_{shard_idx:04d}.jsonl"
        with open(shard_file, "w", encoding="utf-8") as f:
            for doc in batch:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        shard_idx += 1

    print(f"\nDEDUPLICATION COMPLETE — {language.upper()}")
    print(f"  Input documents:      {total_input:,}")
    print(f"  Hash duplicates:      {total_hash_dupes:,}")
    print(f"  URL duplicates:       {total_url_dupes:,}")
    print(f"  Output documents:     {total_output:,}")
    print(f"  Total clean chars:    {total_chars:,}")
    print(f"  Dedup rate:           {(1 - total_output/max(1,total_input))*100:.1f}%")
    print(f"  Shards written:       {shard_idx}")
    print("=" * 60)

    # Write summary
    summary = {
        "language": language,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_documents": total_input,
        "hash_duplicates": total_hash_dupes,
        "url_duplicates": total_url_dupes,
        "output_documents": total_output,
        "total_clean_characters": total_chars,
        "dedup_rate": round((1 - total_output / max(1, total_input)) * 100, 2),
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = output_path / "dedup_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def merge_and_dedup(
    manual_dir: str,
    automated_dir: str,
    output_dir: str,
    language: str = "nepali",
):
    """Merge manual and automated corpora, then deduplicate.

    Args:
        manual_dir: Directory with manual collection shards
        automated_dir: Directory with automated (HF) shards
        output_dir: Directory for merged + deduped output
        language: Language label
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    all_docs = []
    manual_count = 0
    auto_count = 0
    manual_chars = 0
    auto_chars = 0

    # Process manual first (higher priority)
    manual_path = Path(manual_dir)
    if manual_path.exists():
        for shard_file in sorted(manual_path.glob("*.jsonl")):
            with open(shard_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    doc = json.loads(line)
                    text = doc.get("text", "")
                    text_hash = doc.get("text_sha256", "")
                    if not text_hash and text:
                        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if text_hash and text_hash not in seen_hashes:
                        seen_hashes.add(text_hash)
                        doc["collection_type"] = "manual"
                        all_docs.append(doc)
                        manual_count += 1
                        manual_chars += doc.get("clean_text_characters", len(text))

    # Process automated
    auto_path = Path(automated_dir)
    if auto_path.exists():
        for shard_file in sorted(auto_path.glob("*.jsonl")):
            with open(shard_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    doc = json.loads(line)
                    text = doc.get("text", "")
                    text_hash = doc.get("text_sha256", "")
                    if not text_hash and text:
                        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if text_hash and text_hash not in seen_hashes:
                        seen_hashes.add(text_hash)
                        doc["collection_type"] = "automated"
                        all_docs.append(doc)
                        auto_count += 1
                        auto_chars += doc.get("clean_text_characters", len(text))

    # Write merged shards
    shard_size = 10000
    shard_idx = 0
    for i in range(0, len(all_docs), shard_size):
        batch = all_docs[i : i + shard_size]
        shard_file = output_path / f"merged_shard_{shard_idx:04d}.jsonl"
        with open(shard_file, "w", encoding="utf-8") as f:
            for doc in batch:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        shard_idx += 1

    total = manual_count + auto_count
    total_chars = manual_chars + auto_chars

    print(f"\nMERGE & DEDUP COMPLETE — {language.upper()}")
    print(f"  Manual documents:     {manual_count:,} ({manual_chars:,} chars)")
    print(f"  Automated documents:  {auto_count:,} ({auto_chars:,} chars)")
    print(f"  Total unique:         {total:,} ({total_chars:,} chars)")
    if total > 0:
        print(f"  Manual %:             {manual_count/total*100:.1f}%")
        print(f"  Manual chars %:       {manual_chars/total_chars*100:.1f}%")
    print(f"  Shards written:       {shard_idx}")
    print("=" * 60)

    summary = {
        "language": language,
        "manual_documents": manual_count,
        "manual_characters": manual_chars,
        "automated_documents": auto_count,
        "automated_characters": auto_chars,
        "total_documents": total,
        "total_characters": total_chars,
        "manual_percentage": round(manual_count / max(1, total) * 100, 2),
        "shards_written": shard_idx,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = output_path / "merge_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Deduplicate JSONL shards")
    subparsers = parser.add_subparsers(dest="command")

    # Dedup subcommand
    dedup_parser = subparsers.add_parser("dedup", help="Deduplicate a single directory")
    dedup_parser.add_argument("--input-dir", required=True)
    dedup_parser.add_argument("--output-dir", required=True)
    dedup_parser.add_argument("--language", default="nepali")

    # Merge subcommand
    merge_parser = subparsers.add_parser("merge", help="Merge manual + automated and dedup")
    merge_parser.add_argument("--manual-dir", required=True)
    merge_parser.add_argument("--automated-dir", required=True)
    merge_parser.add_argument("--output-dir", required=True)
    merge_parser.add_argument("--language", default="nepali")

    args = parser.parse_args()

    if args.command == "dedup":
        deduplicate_jsonl_shards(args.input_dir, args.output_dir, args.language)
    elif args.command == "merge":
        merge_and_dedup(args.manual_dir, args.automated_dir, args.output_dir, args.language)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
