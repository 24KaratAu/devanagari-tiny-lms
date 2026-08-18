"""
Train/Val/Test Splitting Pipeline

Creates document-level train/val/test splits from deduplicated JSONL shards.
Ensures no data leakage across splits. Outputs split manifests and data files.

Usage:
    python -m src.data.split.create_splits \
        --input-dir data/nepali/processed/merged/ \
        --output-dir data/nepali/splits/ \
        --language nepali \
        --train-ratio 0.90 --val-ratio 0.05 --test-ratio 0.05
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter


def create_splits(
    input_dir: str,
    output_dir: str,
    language: str = "nepali",
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
):
    """Create train/val/test splits from JSONL shards.

    Uses document-level splitting with stratification by source_type
    to maintain source distribution across splits.

    Args:
        input_dir: Directory containing JSONL shard files
        output_dir: Directory for split output files
        language: Language label
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        seed: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all documents
    print(f"Loading documents from {input_dir}...")
    all_docs = []
    shard_files = sorted(input_path.glob("*.jsonl"))

    for sf in shard_files:
        with open(sf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        doc = json.loads(line)
                        all_docs.append(doc)
                    except json.JSONDecodeError:
                        continue

    if not all_docs:
        print("No documents found!")
        return

    print(f"Total documents loaded: {len(all_docs):,}")

    # Group by source_type for stratified splitting
    source_groups = {}
    for doc in all_docs:
        source = doc.get("source_type", "unknown")
        if source not in source_groups:
            source_groups[source] = []
        source_groups[source].append(doc)

    print(f"\nSource distribution:")
    for source, docs in sorted(source_groups.items()):
        print(f"  {source}: {len(docs):,} docs")

    # Stratified split
    random.seed(seed)
    train_docs = []
    val_docs = []
    test_docs = []

    for source, docs in source_groups.items():
        random.shuffle(docs)
        n = len(docs)
        n_val = max(1, int(n * val_ratio)) if n > 2 else 0
        n_test = max(1, int(n * test_ratio)) if n > 2 else 0
        n_train = n - n_val - n_test

        train_docs.extend(docs[:n_train])
        val_docs.extend(docs[n_train:n_train + n_val])
        test_docs.extend(docs[n_train + n_val:])

    # Shuffle within each split
    random.shuffle(train_docs)
    random.shuffle(val_docs)
    random.shuffle(test_docs)

    # Write splits as JSONL files
    splits = {
        "train": train_docs,
        "val": val_docs,
        "test": test_docs,
    }

    for split_name, docs in splits.items():
        split_file = output_path / f"{split_name}.jsonl"
        with open(split_file, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Also write plain text files for tokenizer training
    for split_name, docs in splits.items():
        text_file = output_path / f"{split_name}.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            for doc in docs:
                text = doc.get("text", "")
                if text.strip():
                    f.write(text.strip() + "\n\n")

    # Compute statistics
    def split_stats(docs, name):
        total_chars = sum(d.get("clean_text_characters", len(d.get("text", ""))) for d in docs)
        source_counts = Counter(d.get("source_type", "unknown") for d in docs)
        collection_counts = Counter(d.get("collection_type", "unknown") for d in docs)
        return {
            "split": name,
            "documents": len(docs),
            "total_characters": total_chars,
            "source_distribution": dict(source_counts),
            "collection_type_distribution": dict(collection_counts),
        }

    stats = {
        "language": language,
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "splits": {
            name: split_stats(docs, name)
            for name, docs in splits.items()
        },
        "total_documents": len(all_docs),
        "total_characters": sum(
            d.get("clean_text_characters", len(d.get("text", ""))) for d in all_docs
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    stats_path = output_path / "split_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SPLIT COMPLETE — {language.upper()}")
    print(f"{'='*60}")
    for name, docs in splits.items():
        total_chars = sum(d.get("clean_text_characters", len(d.get("text", ""))) for d in docs)
        manual = sum(1 for d in docs if d.get("collection_type") == "manual")
        auto = len(docs) - manual
        print(
            f"  {name:6s}: {len(docs):>8,} docs | "
            f"{total_chars:>14,} chars | "
            f"manual: {manual:,} / auto: {auto:,}"
        )
    print(f"{'='*60}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--language", default="nepali")
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    create_splits(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        language=args.language,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
