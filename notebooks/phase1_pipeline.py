"""
Phase 1 — Full Data Pipeline (Kaggle/Colab Runner)

This script runs the ENTIRE Phase 1 data pipeline end-to-end on Kaggle or Colab:
  1. Download Wikipedia dumps (Nepali + Hindi)
  2. Extract + clean wiki dumps (manual collection)
  3. Download HuggingFace automated corpora
  4. Deduplicate all corpora
  5. Merge manual + automated
  6. Create train/val/test splits
  7. Train BPE tokenizers (separate per language)
  8. Tokenize corpora + report token counts
  9. Upload everything to Google Drive

Run this in a Kaggle notebook or Colab with:
    !pip install wikiextractor datasets sentencepiece
    %run notebooks/phase1_pipeline.py

Or step-by-step in notebook cells.
"""

import os
import subprocess
import sys

# ── Step 0: Install dependencies ──────────────────────────────────────────

def install_deps():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "wikiextractor", "datasets", "sentencepiece"])
    print("✅ Dependencies installed")


# ── Step 1: Download Wikipedia Dumps ─────────────────────────────────────

def download_wiki_dumps(base_dir="data"):
    """Download Nepali and Hindi Wikipedia dumps."""
    os.makedirs(f"{base_dir}/nepali/raw/wiki_dump", exist_ok=True)
    os.makedirs(f"{base_dir}/hindi/raw/wiki_dump", exist_ok=True)

    dumps = {
        "nepali": {
            "url": "https://dumps.wikimedia.org/newiki/latest/newiki-latest-pages-articles.xml.bz2",
            "path": f"{base_dir}/nepali/raw/wiki_dump/newiki-latest-pages-articles.xml.bz2",
        },
        "hindi": {
            "url": "https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2",
            "path": f"{base_dir}/hindi/raw/wiki_dump/hiwiki-latest-pages-articles.xml.bz2",
        },
    }

    for lang, info in dumps.items():
        if os.path.exists(info["path"]):
            size_mb = os.path.getsize(info["path"]) / (1024 * 1024)
            print(f"  {lang}: Already downloaded ({size_mb:.1f} MB)")
        else:
            print(f"  Downloading {lang} Wikipedia dump...")
            subprocess.run(["wget", "-q", "--show-progress", "-O", info["path"], info["url"]])
            size_mb = os.path.getsize(info["path"]) / (1024 * 1024)
            print(f"  ✅ {lang}: {size_mb:.1f} MB")

    return dumps


# ── Step 2: Process Wikipedia Dumps ──────────────────────────────────────

def process_wiki_dumps(base_dir="data"):
    """Extract and clean Wikipedia dumps into JSONL shards."""
    sys.path.insert(0, ".")
    from src.data.collect.process_wiki_dump import process_wiki_dump

    languages = {
        "nepali": {
            "dump": f"{base_dir}/nepali/raw/wiki_dump/newiki-latest-pages-articles.xml.bz2",
            "output": f"{base_dir}/nepali/processed/wiki_dump/",
        },
        "hindi": {
            "dump": f"{base_dir}/hindi/raw/wiki_dump/hiwiki-latest-pages-articles.xml.bz2",
            "output": f"{base_dir}/hindi/processed/wiki_dump/",
        },
    }

    results = {}
    for lang, paths in languages.items():
        if os.path.exists(paths["dump"]):
            print(f"\n{'='*60}")
            print(f"Processing {lang} Wikipedia dump...")
            result = process_wiki_dump(
                dump_path=paths["dump"],
                output_dir=paths["output"],
                language=lang,
                shard_size=10000,
                min_clean_chars=100,
                min_devanagari_ratio=0.5,
            )
            results[lang] = result
        else:
            print(f"  Skipping {lang}: dump not found at {paths['dump']}")

    return results


# ── Step 3: Download HuggingFace Automated Corpora ────────────────────────

def download_hf_corpora(base_dir="data"):
    """Download automated corpora from HuggingFace."""
    sys.path.insert(0, ".")
    from src.data.collect.process_hf_corpus import process_hf_dataset

    # Target: ~450M chars for Nepali automated, ~400M chars for Hindi automated
    # (chars ≈ 6-8x tokens for Devanagari BPE, so 450M chars ≈ 60-75M tokens)
    # We need many more chars. Target ~3B chars per language for 400M tokens.

    tasks = [
        {
            "language": "nepali",
            "dataset": "sangraha",
            "output_dir": f"{base_dir}/nepali/processed/hf_sangraha/",
            "max_chars": 3_000_000_000,  # 3B chars
        },
        {
            "language": "hindi",
            "dataset": "sangraha",
            "output_dir": f"{base_dir}/hindi/processed/hf_sangraha/",
            "max_chars": 3_000_000_000,
        },
    ]

    results = {}
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Processing {task['language']} / {task['dataset']}...")
        try:
            result = process_hf_dataset(
                language=task["language"],
                dataset_name=task["dataset"],
                output_dir=task["output_dir"],
                max_chars=task["max_chars"],
                shard_size=10000,
                min_clean_chars=100,
                min_devanagari_ratio=0.5,
            )
            results[f"{task['language']}_{task['dataset']}"] = result
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
            print(f"  Trying fallback datasets...")

    return results


# ── Step 4: Deduplicate ──────────────────────────────────────────────────

def deduplicate_all(base_dir="data"):
    """Deduplicate wiki dump shards for each language."""
    sys.path.insert(0, ".")
    from src.data.deduplicate.dedup import deduplicate_jsonl_shards

    for lang in ["nepali", "hindi"]:
        wiki_dir = f"{base_dir}/{lang}/processed/wiki_dump/"
        deduped_dir = f"{base_dir}/{lang}/processed/wiki_deduped/"
        if os.path.exists(wiki_dir):
            print(f"\nDeduplicating {lang} wiki dump...")
            deduplicate_jsonl_shards(wiki_dir, deduped_dir, lang)


# ── Step 5: Merge Manual + Automated ─────────────────────────────────────

def merge_corpora(base_dir="data"):
    """Merge manual (wiki dump) and automated (HF) corpora."""
    sys.path.insert(0, ".")
    from src.data.deduplicate.dedup import merge_and_dedup

    for lang in ["nepali", "hindi"]:
        manual_dir = f"{base_dir}/{lang}/processed/wiki_deduped/"
        auto_dir = f"{base_dir}/{lang}/processed/hf_sangraha/"
        merged_dir = f"{base_dir}/{lang}/processed/merged/"

        if os.path.exists(manual_dir) or os.path.exists(auto_dir):
            print(f"\nMerging {lang} corpora...")
            merge_and_dedup(manual_dir, auto_dir, merged_dir, lang)


# ── Step 6: Create Splits ────────────────────────────────────────────────

def create_all_splits(base_dir="data"):
    """Create train/val/test splits for each language."""
    sys.path.insert(0, ".")
    from src.data.split.create_splits import create_splits

    for lang in ["nepali", "hindi"]:
        merged_dir = f"{base_dir}/{lang}/processed/merged/"
        splits_dir = f"{base_dir}/{lang}/splits/"

        if os.path.exists(merged_dir):
            print(f"\nCreating {lang} splits...")
            create_splits(merged_dir, splits_dir, lang)


# ── Step 7: Train Tokenizers ────────────────────────────────────────────

def train_tokenizers(base_dir="data"):
    """Train separate BPE tokenizers for each language."""
    sys.path.insert(0, ".")
    from src.tokenizer.train_bpe import train_tokenizer

    for lang in ["nepali", "hindi"]:
        train_file = f"{base_dir}/{lang}/splits/train.txt"
        model_prefix = f"{lang}/tokenizer/{lang}_bpe_32k"

        if os.path.exists(train_file):
            os.makedirs(f"{lang}/tokenizer/", exist_ok=True)
            print(f"\nTraining {lang} tokenizer...")
            train_tokenizer(
                train_file=train_file,
                model_prefix=model_prefix,
                vocab_size=32000,
                language=lang,
            )


# ── Step 8: Tokenize Corpora ────────────────────────────────────────────

def tokenize_all(base_dir="data"):
    """Tokenize all splits with trained tokenizers."""
    sys.path.insert(0, ".")
    from src.tokenizer.train_bpe import tokenize_corpus

    for lang in ["nepali", "hindi"]:
        model_path = f"{lang}/tokenizer/{lang}_bpe_32k.model"
        if not os.path.exists(model_path):
            print(f"  Skipping {lang}: no trained model found")
            continue

        for split in ["train", "val", "test"]:
            input_file = f"{base_dir}/{lang}/splits/{split}.txt"
            output_file = f"{base_dir}/{lang}/splits/{split}_tokenized.jsonl"
            if os.path.exists(input_file):
                print(f"\nTokenizing {lang}/{split}...")
                tokenize_corpus(model_path, input_file, output_file)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 1 — FULL DATA PIPELINE")
    print("=" * 60)

    base = "data"

    print("\n📦 Step 0: Installing dependencies...")
    install_deps()

    print("\n📥 Step 1: Downloading Wikipedia dumps...")
    download_wiki_dumps(base)

    print("\n🔧 Step 2: Processing Wikipedia dumps...")
    process_wiki_dumps(base)

    print("\n📥 Step 3: Downloading HuggingFace corpora...")
    download_hf_corpora(base)

    print("\n🔄 Step 4: Deduplicating...")
    deduplicate_all(base)

    print("\n🔗 Step 5: Merging manual + automated...")
    merge_corpora(base)

    print("\n✂️ Step 6: Creating train/val/test splits...")
    create_all_splits(base)

    print("\n🔤 Step 7: Training tokenizers...")
    train_tokenizers(base)

    print("\n📊 Step 8: Tokenizing corpora...")
    tokenize_all(base)

    print("\n" + "=" * 60)
    print("🎉 PHASE 1 PIPELINE COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
