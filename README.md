[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# Monolingual Transformer Language Models from Scratch: Hindi & Nepali
**Language Models and Agents (CL3-410) — Monsoon 2026**  
*Individual Project: Building, Training, and Evaluating Independent Monolingual LMs*

---

## Project Overview

This project builds two completely independent, from-scratch decoder-only Transformer language models (~25M parameters each) for two Indian languages sharing the Devanagari script:
* **Model H (Higher-Resource)**: **Hindi** (`hi`)
* **Model L (Lower-Resource)**: **Nepali** (`ne`)

### Strict Monolingual Constraints
* **Zero Shared Artifacts**: Each model has its own distinct corpus, tokenizer vocabulary, embedding matrix, and model weights.
* **No Pretrained Models or Tokenizers**: Built strictly from scratch using primitive PyTorch layers (custom Multi-Head Attention, Positional Encodings, and Causal Masking). No Hugging Face `Transformer` model classes.
* **Token Targets**: ~500M tokens per language monolingual corpus, adhering to the at least 20% manual collection and at most 80% automated collection requirement.

---

## Repository Structure

```text
.
├── configs/                         # Model & training configurations
├── data/                            # Local corpus staging & manifests
│   ├── hindi/
│   │   ├── manifests/               # Collection & split manifests
│   │   ├── processed/               # Cleaned JSONL shards
│   │   └── splits/                  # train / val / test splits & tokenized data
│   └── nepali/
│       ├── manifests/               # Process-safe collector lock & manifests
│       ├── processed/               # Cleaned web & wiki shards
│       ├── sources/                 # Seed URLs by category (Gov, News, Lit, Wiki)
│       └── splits/                  # train / val / test splits
├── notebooks/                       # End-to-end Colab execution pipelines
│   ├── collect_hindi_colab.ipynb    # Hindi automated + manual collection & tokenizer training
│   ├── collect_nepali_colab.ipynb   # Nepali automated + manual collection & tokenizer training
│   └── phase1_pipeline.py           # Standalone modular pipeline runner
├── report/                          # Phase reports, metrics, plots, and analysis
├── scripts/                         # Audit, concurrency, and validation test suites
│   ├── audit_batch1_checkpoint.py   # 1,000-doc verification & Drive sharding audit
│   ├── audit_existing_corpus_language.py
│   ├── expand_source_urls.py        # Recursive category crawler
│   └── test_collector_concurrency.py
├── src/                             # Core modular package
│   ├── data/
│   │   ├── clean/                   # Devanagari cleaning & language ID
│   │   │   ├── clean_nepali.py      # Table stripping & citation cleaner
│   │   │   ├── nepali_lang_id.py    # Composite lexical/script language detector
│   │   │   └── migrate_manifest_schema.py
│   │   ├── collect/                 # Collection engines
│   │   │   ├── collect_manual_web.py    # Process-safe Trafilatura crawler
│   │   │   ├── collect_wikimedia_api.py # Wikimedia REST/Action API collector
│   │   │   ├── process_hf_corpus.py     # Automated Hugging Face extractor
│   │   │   └── process_wiki_dump.py     # Fast Wikipedia XML bz2 parser
│   │   ├── deduplicate/             # Content & URL deduplication
│   │   │   └── dedup.py             # SHA-256 exact dedup & merger
│   │   ├── shard_manager.py         # Google Drive rclone packaging & verification
│   │   ├── storage_guard.py         # Disk quota safety monitoring
│   │   └── split/                   # Stratified train/val/test split generator
│   │       └── create_splits.py
│   └── tokenizer/                   # Custom tokenizer training
│       └── train_bpe.py             # SentencePiece BPE trainer (32k vocab)
├── doubts-answers.md                # Official course TA clarifications
└── README.md
```

---

## The Data Collection and Engineering Journey

Building a ~500M token corpus per language on student-tier compute (free Google Colab GPUs/CPUs) required significant architectural iteration. Below is the technical log of the engineering challenges encountered and how our pipeline evolved into a zero-RAM, crash-resilient streaming system.

### 1. The Naive Multi-Source Pipeline and Its Bottlenecks
* **Initial Plan**: Ingest web data from multiple Hugging Face datasets (`ai4bharat/sangraha`, `uonlp/CulturaX`, `cc100`, `oscar`) using `datasets.load_dataset(..., streaming=True)` alongside local web scraping.
* **The Reality**: 
  1. Multilingual web corpora contained significant English boilerplate, navigation junk, and uncurated machine translation.
  2. The `datasets` library streaming engine suffered from erratic socket timeouts, authentication friction on Colab, and internal Arrow buffer caching that caused memory usage to creep upward indefinitely.

### 2. Preserving the 20% Manual Requirement (Avoiding Wikipedia Overlap)
* **The Conflict**: Per course rules, Wikipedia XML dumps processed and cleaned in-house count as manual collection (at least 20%). However, generic web crawls from automated datasets often scrape Wikipedia pages. If automated web scraping ran first, exact hash-deduplication would discard our manual Wikipedia dump articles.
* **The Solution**: Re-ordered collection priorities to ingest non-web automated data (PDF/ASR transcripts from Sangraha) first, followed by the complete manual Wikipedia dump, ensuring 100% of our manual articles were preserved without cross-source collision.

### 3. Google Colab Disconnections and Stateful Drive Checkpointing
* **The Challenge**: Standard Colab sessions disconnect or reset after idle periods, instantly wiping local disk storage (`/content/`). A single 3-hour interrupted run meant starting from scratch.
* **The Solution**: Transformed the pipeline into a fully stateful, crash-safe architecture:
  * JSONL shards flush directly into Google Drive (`DRIVE_BASE/cleaned/...`).
  * On reconnect/restart, the notebook stream-reads existing shards, reconstitutes the deduplication hash set, tallies current token progress, and resumes execution without re-downloading or duplicating records.

### 4. Memory Triage: 64-bit Integer Hashing and Zero-RAM Splitting
* **The Challenge**: Tracking millions of unique 64-character SHA-256 strings in a Python `set` consumed several gigabytes of RAM, triggering Colab Out-Of-Memory (OOM) process termination. Similarly, loading the entire merged corpus into memory to split `train/val/test` caused immediate crashes.
* **The Solution**:
  * **64-bit Integer Hashing**: Stored only the first 16 hex characters as 64-bit integers (`int(sha256[:16], 16)`), reducing dedup memory consumption by **~75%**.
  * **Zero-RAM Streaming Splitting**: Replaced in-memory split logic with a streaming file router: each incoming document is assigned via a pseudo-random seed directly to `train.txt`, `val.txt`, or `test.txt` disk handles.
  * **SentencePiece Capping**: Configured `input_sentence_size=500_000` to train high-fertility BPE models without loading multi-gigabyte text files into SentencePiece memory.

### 5. Ditching the `datasets` Library for Pure PyArrow Streaming
* **The Challenge**: Even with streaming enabled, `load_dataset('ai4bharat/sangraha', ...)` periodically hung when establishing HTTP stream connections to Hugging Face parquet shards.
* **The Solution**: Replaced Hugging Face's wrapper with direct Python `urllib.request` + `pyarrow`. The notebook queries the Hugging Face API for the parquet file listing, downloads one ~340MB file at a time to local disk, processes it, flushes shards to Drive, deletes the temporary file, and triggers `gc.collect()`.

### 6. The Wikipedia XML Memory Leak (`root.clear()`)
* **The Challenge**: During streaming parse of Wikipedia dumps (`newiki-latest-pages-articles.xml.bz2`), Python's `xml.etree.ElementTree.iterparse` accumulated parent DOM references, leaking memory from ~200MB up to 10GB+ before crashing.
* **The Solution**: Added `root.clear()` alongside `elem.clear()` on every processed `<page>` element, keeping XML parsing memory strictly flat under **300MB RAM** across all 50,000+ Wikipedia articles.

### 7. The `data-0.parquet` Trap and Micro-Batching
* **The Challenge**: In the Sangraha repository, `data-0.parquet` is an uncompressed ~8.6GB behemoth that instantly exhausts local Colab disk space, whereas subsequent shards (`data-1.parquet`, `data-2.parquet`, etc.) are neat ~340MB files.
* **The Solution**: Explicitly filtered out `data-0.parquet` and processed the ~340MB files using `pyarrow.parquet.ParquetFile.iter_batches(batch_size=5000)`. Only ~20MB of records are ever held in memory at any millisecond.

### 8. Final High-Yield Convergence (~400M Tokens in ~25 Minutes)
* **The Breakthrough**: Originally, we filtered Sangraha strictly for PDF and ASR transcripts. However, because each 340MB parquet chunk contained mixed document types, discarding web text meant each file yielded only 2–3M tokens, requiring over 50 files to hit our target.
* **The Final Architecture**: Sangraha Verified documents are human-curated and clean. By accepting all verified documents (PDF, ASR, and Web), each 340MB file yielded **35M+ clean tokens**. The entire 400M automated token target is now reached in just **11–12 parquet files (~25 minutes total runtime)**.

---

## Dataset and Cleaning Specifications

### Quality Filters Applied
1. **Devanagari Script Purity**: Every document must meet a minimum Devanagari ratio of at least 0.50 (Devanagari Unicode block `\u0900`–`\u097F`).
2. **Length Threshold**: Minimum clean document length of at least 100 characters.
3. **Markup and Table Removal**: MediaWiki syntax, XML tables, citations (`[1]`), and external navigation links stripped via regex pipelines.
4. **Composite Language Identification**: Heuristic and lexical signature filtering (differentiating Nepali case markers like `लाई`, `मा`, `बाट` from Hindi markers `है`, `में`, `पर`) to reject cross-lingual contamination.
5. **Exact Deduplication**: SHA-256 content hashing across all manual and automated shards.

### Split Stratification & Corpus Accounting
All corpora are partitioned using a reproducible document-level split (Seed `42`):
* **Train Set (90%)**: Used for pretraining language models and training tokenizers (~439.6M tokens).
* **Validation Set (5%)**: Held-out set for loss, perplexity (PPL), and Bits-Per-Byte (BPB) tracking (~24.4M tokens).
* **Test Set (5%)**: Final evaluation split for generation quality (BLEU, chrF, ROUGE-L) and attention map analysis (~24.5M tokens).

#### Corpus Scale & Balance (Nepali — Model L):
* **Manual Data Shards (48 Shards — 1.28 GiB)**: ~253.6M tokens (~8.65M tokens from Wikipedia dumps + ~245M tokens from live news crawls: *Ratopati*, *Setopati*, *Gorkhapatra*). Comfortably exceeds the $\ge 20\%$ manual floor.
* **Automated Data Shards (248 Shards — 6.26 GiB)**: ~1.25B tokens from AI4Bharat Sangraha Verified.
* **Total Cleaned Storage on Google Drive**: **296 shards (7.54 GiB, ~1.50B tokens)**.
* **Active Pretraining Dataset**: **488,502,833 tokens (~488.5M tokens)** partitioned into the 90/5/5 splits.

---

## Tokenizer Construction

Trained separate **SentencePiece Byte-Pair Encoding (BPE)** tokenizers from scratch for each language on the training split:
* **Model Type**: BPE (`byte_fallback=True` to prevent out-of-vocabulary `<unk>` tokens on rare characters/emojis).
* **Vocabulary Size**: `32,000` tokens per language.
* **Character Coverage**: `0.9995` (tailored for Devanagari combining characters, nuktas, and virama `्`).
* **Special Tokens**: `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`.
* **Script Preservation**: `split_by_unicode_script=True`, `split_by_whitespace=True`, `split_digits=True`.

---

## How to Reproduce

### Option A: Run on Google Colab (Recommended)
1. Open [`notebooks/collect_nepali_colab.ipynb`](notebooks/collect_nepali_colab.ipynb) or [`notebooks/collect_hindi_colab.ipynb`](notebooks/collect_hindi_colab.ipynb) in Google Colab.
2. Mount your Google Drive in Cell 2.
3. Run all cells. The notebook will:
   * Download and stream Sangraha verified files with instant early stop at 400M tokens.
   * Download and parse the full Wikipedia XML dump (manual).
   * Merge and perform exact SHA-256 deduplication.
   * Create 90/5/5 train/val/test splits.
   * Train the 32k SentencePiece BPE tokenizer and save all artifacts to Drive.

### Option B: Local CLI Execution
```bash
# 1. Install dependencies
pip install -r requirements.txt  # sentencepiece, pyarrow, pandas, trafilatura, requests

# 2. Process Wikipedia XML Dump (Manual Collection)
python -m src.data.collect.process_wiki_dump \
    --language nepali \
    --dump-path data/nepali/raw/wiki_dump/newiki-latest-pages-articles.xml.bz2 \
    --output-dir data/nepali/processed/wiki_dump/

# 3. Process Automated Sangraha Shards
python -m src.data.collect.process_hf_corpus \
    --language nepali \
    --dataset sangraha \
    --output-dir data/nepali/processed/hf_sangraha/

# 4. Merge & Deduplicate
python -m src.data.deduplicate.dedup merge \
    --manual-dir data/nepali/processed/wiki_dump/ \
    --automated-dir data/nepali/processed/hf_sangraha/ \
    --output-dir data/nepali/processed/merged/ \
    --language nepali

# 5. Create Train/Val/Test Splits
python -m src.data.split.create_splits \
    --input-dir data/nepali/processed/merged/ \
    --output-dir data/nepali/splits/ \
    --language nepali

# 6. Train SentencePiece BPE Tokenizer
python -m src.tokenizer.train_bpe train \
    --train-file data/nepali/splits/train.txt \
    --model-prefix nepali/tokenizer/nepali_bpe_32k \
    --vocab-size 32000 \
    --language nepali
```

---

## Google Drive Artifacts and Checkpoints

All large raw archives, processed JSONL shards, split files, and trained tokenizer models are hosted on Google Drive:

| Artifact Description | Language | Link |
| :--- | :---: | :---: |
| **Complete Cleaned Dataset (JSONL Shards)** | Nepali (`ne`) | *[Google Drive Link Placeholder]* |
| **Complete Cleaned Dataset (JSONL Shards)** | Hindi (`hi`) | *[Google Drive Link Placeholder]* |
| **Train / Val / Test Splits** | Nepali (`ne`) | *[Google Drive Link Placeholder]* |
| **Train / Val / Test Splits** | Hindi (`hi`) | *[Google Drive Link Placeholder]* |
| **SentencePiece BPE Model & Vocab (`32k`)** | Nepali (`ne`) | *[Google Drive Link Placeholder]* |
| **SentencePiece BPE Model & Vocab (`32k`)** | Hindi (`hi`) | *[Google Drive Link Placeholder]* |
| **Pretrained Model Checkpoints (Phase 2)** | Model L / Model H | *Coming in Phase 2* |
| **Finetuned Model Checkpoints (Phase 3)** | Model L / Model H | *Coming in Phase 3* |

---

*Authored by 24KaratAu for LMA Monsoon 2026.*
