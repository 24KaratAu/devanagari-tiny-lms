"""
HuggingFace Corpus Processor

Downloads and processes public HuggingFace datasets for the automated (80%)
portion of the corpus. Supports streaming mode to handle large datasets
without exhausting disk space.

Supported datasets:
  - ai4bharat/sangraha (Nepali, Hindi, and other Indian languages)
  - oscar-corpus/OSCAR-2301 (multilingual web corpus)
  - uonlp/CulturaX (cleaned Common Crawl)
  - cc100 (Common Crawl monolingual)

Usage:
    python -m src.data.collect.process_hf_corpus \
        --language nepali \
        --dataset sangraha \
        --output-dir data/nepali/processed/hf_corpus/ \
        --max-tokens 450000000 \
        --shard-size 10000
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime, timezone

# Dataset configurations
DATASET_CONFIGS = {
    "sangraha": {
        "path": "ai4bharat/sangraha",
        "nepali": {"name": "verified", "lang_filter": "npi", "text_field": "text"},
        "hindi": {"name": "verified", "lang_filter": "hin", "text_field": "text"},
    },
    "culturax": {
        "path": "uonlp/CulturaX",
        "nepali": {"name": "ne", "lang_filter": None, "text_field": "text"},
        "hindi": {"name": "hi", "lang_filter": None, "text_field": "text"},
    },
    "cc100": {
        "path": "cc100",
        "nepali": {"name": "ne", "lang_filter": None, "text_field": "text"},
        "hindi": {"name": "hi", "lang_filter": None, "text_field": "text"},
    },
    "oscar": {
        "path": "oscar-corpus/OSCAR-2301",
        "nepali": {"name": "ne", "lang_filter": None, "text_field": "text"},
        "hindi": {"name": "hi", "lang_filter": None, "text_field": "text"},
    },
}


def devanagari_ratio(text: str) -> float:
    """Calculate ratio of Devanagari characters."""
    if not text:
        return 0.0
    dev = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    return dev / len(text)


def clean_web_text(text: str) -> str:
    """Clean web-crawled text."""
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    # Remove HTML entities
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    # Remove excessive punctuation
    text = re.sub(r"[!?]{3,}", "!", text)
    text = re.sub(r"\.{4,}", "...", text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def process_hf_dataset(
    language: str,
    dataset_name: str,
    output_dir: str,
    max_docs: int = 0,
    max_chars: int = 0,
    shard_size: int = 10000,
    min_clean_chars: int = 100,
    min_devanagari_ratio: float = 0.5,
):
    """Download and process a HuggingFace dataset into JSONL shards.

    Args:
        language: "nepali" or "hindi"
        dataset_name: Key from DATASET_CONFIGS
        output_dir: Directory for output shards
        max_docs: Maximum documents to process (0 = unlimited)
        max_chars: Maximum total characters (0 = unlimited)
        shard_size: Documents per shard
        min_clean_chars: Minimum cleaned character count
        min_devanagari_ratio: Minimum Devanagari ratio
    """
    from datasets import load_dataset

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]
    lang_config = config.get(language)
    if not lang_config:
        raise ValueError(f"No config for language '{language}' in dataset '{dataset_name}'")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading {dataset_name} for {language} (streaming mode)...")
    print(f"  HF path: {config['path']}, config: {lang_config['name']}")

    ds = load_dataset(
        config["path"],
        lang_config["name"],
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    text_field = lang_config["text_field"]
    lang_filter = lang_config.get("lang_filter")

    shard_idx = 0
    shard_docs = []
    total_accepted = 0
    total_rejected = 0
    total_chars = 0
    seen_hashes = set()

    print(f"Processing... (max_docs={max_docs or 'unlimited'}, max_chars={max_chars or 'unlimited'})")
    print("=" * 60)

    for example in ds:
        # Language filter (for multi-language datasets like Sangraha)
        if lang_filter:
            ex_lang = example.get("language", "") or example.get("lang", "")
            if ex_lang != lang_filter:
                continue

        text = example.get(text_field, "")
        if not text or len(text.strip()) < min_clean_chars:
            total_rejected += 1
            continue

        # Clean
        cleaned = clean_web_text(text)
        if len(cleaned) < min_clean_chars:
            total_rejected += 1
            continue

        # Devanagari check
        dev_r = devanagari_ratio(cleaned)
        if dev_r < min_devanagari_ratio:
            total_rejected += 1
            continue

        # Exact dedup via hash
        text_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        if text_hash in seen_hashes:
            total_rejected += 1
            continue
        seen_hashes.add(text_hash)

        doc_id = hashlib.md5(text_hash.encode()).hexdigest()[:16]
        record = {
            "doc_id": doc_id,
            "source": f"{dataset_name}",
            "source_type": f"hf_{dataset_name}",
            "collection_type": "automated",
            "language": language,
            "text": cleaned,
            "clean_text_characters": len(cleaned),
            "devanagari_ratio": round(dev_r, 4),
            "text_sha256": text_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        shard_docs.append(record)
        total_accepted += 1
        total_chars += len(cleaned)

        # Progress reporting
        if total_accepted % 10000 == 0:
            print(
                f"  [{total_accepted:,} accepted] "
                f"Rejected: {total_rejected:,} | "
                f"Chars: {total_chars:,} | "
                f"Dedup cache: {len(seen_hashes):,}"
            )

        # Write shard when full
        if len(shard_docs) >= shard_size:
            shard_file = output_path / f"{dataset_name}_shard_{shard_idx:04d}.jsonl"
            with open(shard_file, "w", encoding="utf-8") as f:
                for doc in shard_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            print(
                f"  OK  Wrote {shard_file.name}: {len(shard_docs):,} docs, "
                f"{sum(d['clean_text_characters'] for d in shard_docs):,} chars"
            )
            shard_docs = []
            shard_idx += 1

        # Check limits
        if max_docs and total_accepted >= max_docs:
            print(f"  Reached max_docs limit: {max_docs}")
            break
        if max_chars and total_chars >= max_chars:
            print(f"  Reached max_chars limit: {max_chars:,}")
            break

    # Write remaining
    if shard_docs:
        shard_file = output_path / f"{dataset_name}_shard_{shard_idx:04d}.jsonl"
        with open(shard_file, "w", encoding="utf-8") as f:
            for doc in shard_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(
            f"  OK  Wrote {shard_file.name}: {len(shard_docs):,} docs, "
            f"{sum(d['clean_text_characters'] for d in shard_docs):,} chars"
        )

    print("\n" + "=" * 60)
    print(f"HF CORPUS PROCESSING COMPLETE — {language.upper()} / {dataset_name}")
    print(f"  Total accepted:          {total_accepted:,}")
    print(f"  Total rejected:          {total_rejected:,}")
    print(f"  Total clean characters:  {total_chars:,}")
    print(f"  Unique documents (hash): {len(seen_hashes):,}")
    print(f"  Shards written:          {shard_idx + 1}")
    print("=" * 60)

    # Write summary
    summary = {
        "language": language,
        "dataset": dataset_name,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "total_clean_characters": total_chars,
        "unique_hashes": len(seen_hashes),
        "shards_written": shard_idx + 1,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = output_path / f"{dataset_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Process HuggingFace datasets into clean JSONL shards")
    parser.add_argument("--language", required=True, choices=["nepali", "hindi"])
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-docs", type=int, default=0, help="Max documents (0=unlimited)")
    parser.add_argument("--max-chars", type=int, default=0, help="Max total characters (0=unlimited)")
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--min-devanagari-ratio", type=float, default=0.5)

    args = parser.parse_args()

    process_hf_dataset(
        language=args.language,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        max_docs=args.max_docs,
        max_chars=args.max_chars,
        shard_size=args.shard_size,
        min_clean_chars=args.min_chars,
        min_devanagari_ratio=args.min_devanagari_ratio,
    )


if __name__ == "__main__":
    main()
