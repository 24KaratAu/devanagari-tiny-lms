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
│   ├── finalize_hindi_corpus.ipynb  # Budget-aware Hindi split + tokenizer retrain
│   ├── finalize_nepali_corpus.ipynb # Budget-aware Nepali split + tokenizer retrain
│   └── phase1_pipeline.py           # Standalone modular pipeline runner
├── tokenizers/                      # Deliverables 6 & 7 — committed, not Drive-only
│   ├── hindi/hindi_bpe_8k.{model,vocab}
│   └── nepali/nepali_bpe_8k.{model,vocab}
├── report/                          # Phase reports, metrics, plots, and analysis
│   ├── phase1_report.md             # Phase 1 report
│   ├── collection_ledger.md         # Every collection stage: commit, script, shards, counts
│   ├── collection_ledger.json       # Machine-readable source for the above
│   └── {hindi,nepali}_tokenizer_benchmark.json
├── scripts/                         # Collection, audit, and validation suites
│   ├── discover_sitemap_urls.py     # robots.txt -> sitemap article-URL discovery
│   ├── collect_manual_web_sitemap.py    # Resumable crawler (--language hindi|nepali)
│   ├── process_hindi_wikimedia_dump.py  # Wikimedia XML dump -> manual shards
│   ├── export_manual_hashes.py      # Dedup registry, streamed from Drive
│   ├── benchmark_tokenizer.py       # Fertility / coverage / vocab sweep on held-out text
│   ├── render_collection_ledger.py  # collection_ledger.json -> collection_ledger.md
│   ├── verify_corpus_claims.py      # Reproduces every measured claim in the report
│   ├── probe_sitemap_availability.py # Which outlets publish crawlable sitemaps
│   ├── audit_hindi_corpus.py        # Exact per-source token accounting from Drive
│   ├── audit_batch1_checkpoint.py   # 1,000-doc verification & Drive sharding audit
│   ├── audit_existing_corpus_language.py
│   ├── expand_source_urls.py        # Recursive category crawler
│   └── test_collector_concurrency.py
├── src/                             # Core modular package
│   ├── data/
│   │   ├── clean/                   # Devanagari cleaning & language ID
│   │   │   ├── clean_hindi_web.py   # Line-level web cleaner & teaser stripper
│   │   │   ├── clean_nepali_web.py  # Nepali marker vocabulary over the same cleaner
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

### 8. The 20% Floor Is a Multiplier, Not an Independent Check
* **The Challenge**: The Hindi corpus stalled below target and the reason was
  structural. A manual share of at least 20% means automated data may be at most
  $4\times$ the manual total, so the **reachable corpus is capped at $5\times$
  the manual token count**. With 88.8M Hindi manual tokens the whole corpus was
  capped at 444M — under the 500M target — while 683M automated tokens sat
  unusable in reserve. No amount of extra automated data could have helped.
  Collecting 161M further manual characters lifted the ceiling to ~699M and the
  delivered corpus to 520.4M tokens at a 26.87% manual share.
* **The Solution**: Size manual collection from the target first
  ($\text{target} / 5$ = minimum manual), and derive the automated budget from
  the manual total at split time as $A = \min(4M,\ T - M)$, so both constraints
  are enforced together rather than one being applied and the other hoped for.

### 9. Guessed URLs vs. Published Sitemaps
* **The Challenge**: The first Hindi news crawler enumerated articles by guessing
  sequential ids (`amarujala.com/india-news/news-id-849999`). Publishers do not
  expose contiguous id spaces; almost every request 404'd and the crawl produced
  13 MB of text.
* **The Solution**: `scripts/discover_sitemap_urls.py` reads each publisher's
  `robots.txt`, follows every `Sitemap:` directive and descends the sitemap index.
  The naive guess of `/sitemap.xml` is itself wrong nearly everywhere — `ndtv.in`
  and `zeenews` return 403, `bhaskar`, `aajtak`, `abplive`, `livehindustan`,
  `news18` and `naidunia` return 404 — yet all of them publish working sitemaps via
  `robots.txt`. Result: **11,112,623 live article URLs across 18 domains in 7.6
  minutes**, reservoir-sampled to a domain-balanced 375,539-URL frontier.

### 10. Devanagari Is Not a Language Filter
* **The Challenge**: Script-ratio filtering cannot separate Hindi from other
  Devanagari languages. `vikaspedia.in` serves ~15 languages from sibling
  subdomains including Nepali (`ne.`), Sanskrit (`sa.`), Maithili and Konkani —
  Nepali being exactly the language of Model L, whose corpus must stay disjoint.
* **The Solution**: Exclude non-Hindi hosts at the frontier, and score every
  crawled document with `hindi_lang_id.py`, which rejects text whose
  Nepali/Sanskrit lexical density exceeds its Hindi density. Auditing the
  pre-existing manual shards the same way returned 97.9–100% Hindi, with a single
  Sanskrit page in the Wikibooks shard.

### 11. Two Transport Traps on the Way to Drive
* **`--bind 0.0.0.0`**: added to work around an intermittent IPv6 route failure
  that aborts rclone during *filesystem construction*, before `--retries` applies.
  Measured effect on this link: throughput collapsed from **1.2 MB/s to 21 KB/s**.
  The fix is an application-level retry loop, not a source-address bind.
* **`RobotFileParser.read()`**: fetches `robots.txt` with urllib's User-Agent,
  which several Hindi portals answer with 403 — and on 403 the parser silently
  switches to *disallow-all*. Left unfixed, this rejected 100% of the frontier
  while reporting a clean run. Fetch `robots.txt` with the crawler's own
  User-Agent and hand the text to `rp.parse()`.

### 12. Asserting on the Plan Instead of the Result
* **The Challenge**: The split notebook staged shards from Drive to local disk with
  a single bulk `cp ... 2>/dev/null`. The FUSE transport dropped partway, the error
  went to the null device, and **2 of 220 automated shards** reached the machine.
  Every downstream check still passed — the budget assert only verified the target
  was *reachable* — so a corpus that was 95.7% manual was written to Drive with no
  warning anywhere. The same silent-truncation bug appeared in the audit and
  dedup-registry scripts, where a short `rclone cat` through a pipe is
  indistinguishable from a short file; it under-reported the corpus by 10,000
  documents.
* **The Solution**: Never move or read a remote shard without verifying the byte
  count against the source, and assert on the *ingested result*, not the plan.
  Staging now copies per file, compares sizes, remounts on `Errno 107` and retries;
  the split refuses to sync unless the automated ingest reached 98% of its budget,
  the total clears 500M, and the manual floor holds.

### 13. Deduplicate the Manual Partition Against Itself
* **The Challenge**: The pipeline deduplicated manual→automated but never
  manual→manual, on the assumption that each collection run had already deduplicated
  itself. It had — but against a registry snapshotted before later shards existed, so
  every run was blind to the ones that followed. Auditing found **2,523 exact
  duplicates / 18.6M characters**, 94% of them because `manual_lit_shard_0030` is a
  re-parse of the same Wikimedia dumps behind `manual_aux_shard_0032`: 2,365 of its
  2,366 documents were already present.
* **The Solution**: Deduplicate the manual partition against itself *before* deriving
  the automated budget from it — otherwise duplicate characters inflate the budget as
  well as the corpus. The freshly written sitemap crawler accounted for 9 of the
  2,523, so the cost of a stale registry falls on whichever run collects last.

### 14. Final High-Yield Convergence (~400M Tokens in ~25 Minutes)
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

#### Corpus Scale & Balance (Hindi — Model H):
* **Manual Data (66 Shards, post-deduplication)**: **240,510 documents /
  479,582,790 clean characters / 139,705,463 tokens** — the Hindi Wikipedia dump
  (73.7M tokens), a sitemap-driven web crawl over 32 Hindi domains (75,387 docs /
  46.9M tokens), the Colab news crawl (12.0M) and Wikibooks/Wikiquote (7.1M).
  Intra-manual deduplication removed 2,523 documents / 18.6M characters, 94% of
  them a redundant re-parse of the same Wikimedia dumps.
* **Automated Data (220 Shards)**: ~682.9M tokens from AI4Bharat Sangraha Verified.
* **Active Pretraining Dataset**: **520,409,408 tokens** across 848,815 documents
  (train 467,905,255 / val 26,069,560 / test 26,434,593), manual share **26.87%**.
  The 139.7M unique manual tokens raise the corpus ceiling from 444M to **~699M**.
* **Chars/token is measured, not assumed**: 3.433 manual, 3.507 automated, 3.484
  on the final corpus. The pipeline re-measures these against the live tokenizer
  before budgeting, because a stale flat constant caused the original shortfall.

#### Corpus Scale & Balance (Nepali — Model L):
* **Manual Data Shards (48 Shards — 1.28 GiB)**: ~253.6M tokens (~8.65M tokens from Wikipedia dumps + ~245M tokens from live news crawls: *Ratopati*, *Setopati*, *Gorkhapatra*). Comfortably exceeds the $\ge 20\%$ manual floor.
* **Automated Data Shards (248 Shards — 6.26 GiB)**: ~1.25B tokens from AI4Bharat Sangraha Verified.
* **Total Cleaned Storage on Google Drive**: **296 shards (7.54 GiB, ~1.50B tokens)**.
* **Active Pretraining Dataset**: **632,711,308 tokens** across 1,367,965 documents
  (train 569,663,061 / val 31,697,868 / test 31,350,379), manual share **21.20%**.
  An audit on 2026-08-21 found the previously reported 488.5M figure counted
  duplicate documents — only **42.7%** of the Nepali manual partition was unique —
  and that the splits held ~16% of the available pool. A 340M-character crawl and a
  re-split with intra-manual deduplication fixed both. See report §3.4, §4.3, §5.1.

---

## Tokenizer Construction

Separate **SentencePiece BPE** tokenizers, trained from scratch on each
language's own training split. No corpus, vocabulary or tokenizer is shared.

* **Model Type**: BPE, `byte_fallback=True` — unknown characters become raw UTF-8 byte tokens instead of `<unk>`, giving a **0.00% unknown-token rate**.
* **Vocabulary Size**: **8,000 (Hindi)**, 32,000 (Nepali). Sizes differ; see the note below.
* **Character Coverage**: `0.9995` (retains Devanagari combining marks, nuktas, virama `्`).
* **Special Tokens**: `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`.
* **Script Preservation**: `split_by_unicode_script=True`, `split_by_whitespace=True`, `split_digits=True`.

### Measured on held-out text

Hindi, on 59,713 validation documents (12.2M characters) never seen in training:

| Statistic | Value |
| :--- | ---: |
| Fertility (tokens/word) | **1.4938** |
| Characters per token | 3.4712 |
| Unknown-token rate | **0.00%** |
| Vocabulary coverage | 97.41% (7,793 of 8,000 pieces observed) |
| Broken grapheme rate | 5.5683% |

### Why BPE, and why 8,000

**Subword over word- or character-level.** Hindi is inflectional
(`विद्यार्थी` → `विद्यार्थियों`), so a word vocabulary would need hundreds of
thousands of entries and still emit `<unk>` on unseen inflections. Character
level avoids that but makes Devanagari sequences 4–5x longer, and attention cost
is quadratic in length. Subword is the only option that fits a 25M budget.

**BPE over Unigram — with a caveat we state openly.** At an equal 8k vocabulary
on the same sample, Unigram actually scored *better*: fertility 1.5613 vs 1.5873
and broken graphemes 6.07% vs 7.97%. BPE was chosen on procedural grounds — it is
the standard for decoder-only pretraining and makes our fertility directly
comparable to the Devanagari literature we benchmark against — not because it
measured better. Report §6.2.2 flags this as the weakest-evidenced decision in
Phase 1 and names the Phase 2 ablation that would settle it.

**Byte fallback** routes unknown characters to raw UTF-8 bytes, which is why the
unknown-token rate is exactly 0.00% at every vocabulary size tested. It decouples
vocabulary size from OOV risk, so the vocabulary need not be inflated defensively.

### Why 8,000 for Hindi

Vocabulary size trades directly against the ~25M parameter budget: with
`d_model = 256` and weight tying, the embedding table costs `V x 256` parameters
— 8.2% of the budget at 8k, but **32.8% at 32k**. A vocabulary sweep on held-out
text (report §6.2) shows 8k -> 32k buys only 0.23 tokens/word of fertility while
vocabulary coverage collapses from 94.31% to 82.39%, leaving ~1.4M embedding
parameters that are never used. The course Q&A also advises a 5–10K vocabulary.

> **Known inconsistency**: Nepali is still at 32,000, predating that guidance. It
> should be retrained at 8k or the choice justified against its parameter cost
> before Phase 2. Recorded in report §6.4 rather than quietly reconciled.

Full benchmark, including the vocabulary sweep and a BPE-vs-Unigram comparison:
[`report/hindi_tokenizer_benchmark.json`](report/hindi_tokenizer_benchmark.json)
(reproduce with `scripts/benchmark_tokenizer.py`).

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

### Option C: Manual Web Collection (Local)

The Hindi manual partition is collected locally and streamed to Drive with
`rclone`, which avoids Colab session drops on a multi-hour crawl. Local disk is
never asked to hold more than one shard.

```bash
# 1. Discover article URLs from publisher sitemaps (~8 min, ~11M URLs)
python scripts/discover_sitemap_urls.py --out data/hindi/sources/frontier.txt

# 2. Snapshot the dedup registry already on Drive
python scripts/export_manual_hashes.py

# 3. Crawl until the character target is met (resumable; Ctrl-C is safe)
python scripts/collect_manual_web_sitemap.py \
    --frontier data/hindi/sources/frontier_balanced.txt \
    --hashes data/hindi/manifests/existing_manual_hashes.txt \
    --target-chars 160000000

# 4. Verify what actually landed on Drive
python scripts/audit_hindi_corpus.py
```

The same scripts serve Nepali via `--language nepali`, pointing at
`data/nepali/sources/` and `gdrive:24KaratAu/nepali/`:

```bash
python scripts/discover_sitemap_urls.py --domains <nepali domains> \
    --out data/nepali/sources/frontier.txt
python scripts/export_manual_hashes.py \
    --remote gdrive:24KaratAu/nepali/cleaned/manual \
    --out data/nepali/manifests/existing_manual_hashes.txt
python scripts/collect_manual_web_sitemap.py --language nepali \
    --frontier data/nepali/sources/frontier_balanced.txt \
    --hashes data/nepali/manifests/existing_manual_hashes.txt \
    --target-chars 340000000
```

Then run the matching notebook on Colab to rebuild the 90/5/5 splits under the
token budget and retrain the 8k BPE tokenizer:
[`finalize_hindi_corpus.ipynb`](notebooks/finalize_hindi_corpus.ipynb) or
[`finalize_nepali_corpus.ipynb`](notebooks/finalize_nepali_corpus.ipynb).

---

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

All large artifacts live in a single shared Google Drive folder:

## 🔗 [**`24KaratAu/` — all datasets, splits and tokenizers**](https://drive.google.com/drive/folders/19EANSjwyIp-z36jwCvKhwlWdcwA1Ohhq)

`https://drive.google.com/drive/folders/19EANSjwyIp-z36jwCvKhwlWdcwA1Ohhq`

Shared as **Anyone with the link → Viewer**, so it opens without an access
request. Every path below is relative to that folder. Report §7 documents which
source produced which file.

| Artifact | Language | Path inside the shared folder | Size |
| :--- | :---: | :--- | ---: |
| Cleaned dataset — automated (JSONL) | `hi` | `hindi/cleaned/automated_nonweb/` | 5.976 GiB |
| Cleaned dataset — manual (JSONL) | `hi` | `hindi/cleaned/manual/` | 1.270 GiB |
| Raw crawled HTML (gzipped per shard) | `hi` | `hindi/raw/manual_web/` | 1.702 GiB |
| Raw Wikimedia dumps (`.xml.bz2`) | `hi` | `hindi/raw/wiki_dump/` | 0.234 GiB |
| Train / val / test splits | `hi` | `hindi/splits/` | 8.834 GiB |
| BPE tokenizer model + vocab (`8k`) | `hi` | `hindi/tokenizer/hindi_bpe_8k.{model,vocab}` | 573 KiB |
| Collection provenance & manifests | `hi` | `hindi/manifests/` | 0.328 GiB |
| Cleaned dataset — automated (JSONL) | `ne` | `nepali/cleaned/automated_nonweb/` | 6.263 GiB |
| Cleaned dataset — manual (JSONL) | `ne` | `nepali/cleaned/manual/` | 1.280 GiB |
| Raw Wikipedia dump + scraped pages | `ne` | `nepali/raw/` | 0.101 GiB |
| Train / val / test splits | `ne` | `nepali/splits/` | 2.659 GiB |
| BPE tokenizer model + vocab (`32k`) | `ne` | `nepali/tokenizer/` | 4 MiB |
| Pretrained checkpoints | H / L | *Coming in Phase 2* | — |
| Finetuned checkpoints | H / L | *Coming in Phase 3* | — |

### Provenance files (Hindi collection)

| File | Path | Contents |
| :--- | :--- | :--- |
| Full URL frontier | `hindi/manifests/frontier_full.txt.gz` | All 11,112,623 sitemap-discovered URLs |
| Balanced frontier | `hindi/manifests/frontier_balanced.txt.gz` | The 375,539-URL crawl input |
| Attempted-URL log | `hindi/manifests/crawl_done_urls.txt.gz` | Every URL the crawler tried |
| Rejection log | `hindi/manifests/crawl_rejected.jsonl.gz` | Per-URL rejection reason |
| Dedup registry | `hindi/manifests/existing_manual_hashes.txt.gz` | Content hashes at crawl time |
| Corpus summary | `hindi/manifests/hindi_corpus_summary.json` | Final split and share figures |

---

*Authored by 24KaratAu for LMA Monsoon 2026.*
