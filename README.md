[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

# Monolingual Transformer Language Models from Scratch: Hindi & Nepali
**Language Models and Agents (CL3-410) — Monsoon 2026**  
*Individual Project: Building, Training, and Evaluating Independent Monolingual LMs*

---

## Phase 1 at a Glance

> **Report** → [`report/phase1_report.md`](report/phase1_report.md) ·
> **Collection ledger** → [`report/collection_ledger.md`](report/collection_ledger.md) ·
> **All data** → [**Google Drive**](https://drive.google.com/drive/folders/19EANSjwyIp-z36jwCvKhwlWdcwA1Ohhq)
> (Anyone with the link → Viewer)

| | Hindi (Model H) | Nepali (Model L) | Requirement |
| :--- | ---: | ---: | :--- |
| Active corpus | **520,409,408 tokens** | **632,711,308 tokens** | ~500M OK  |
| Manual share | **26.87%** | **21.20%** | ≥20% OK  |
| Documents | 848,815 | 1,367,965 | — |
| Tokenizer | `hindi_bpe_8k` | `nepali_bpe_8k` | from scratch, not shared OK  |
| Fertility (tok/word) | 1.4938 | 1.5566 | >1 (sub-word) OK  |
| Unknown-token rate | 0.00% | 0.00% | — |
| Shared documents between corpora | 0 | 0 | none allowed OK  |

**Tokenizer files are committed**, not Drive-only:
[`tokenizers/hindi/`](tokenizers/hindi) · [`tokenizers/nepali/`](tokenizers/nepali)

**Every measured claim in the report is reproducible** — report §7.6 maps each
figure to the command that regenerates it, e.g.:

```bash
python scripts/verify_corpus_claims.py cross-corpus     # corpora share no documents
python scripts/verify_corpus_claims.py uniqueness --language nepali
python scripts/benchmark_tokenizer.py --language hindi --model tokenizers/hindi/hindi_bpe_8k.model ...
```

---

## Phase 2 at a Glance

> **Report** → [`report/phase2_report.md`](report/phase2_report.md) ·
> **Model H** → [`hindi/`](hindi) · **Model L** → [`nepali/`](nepali) ·
> **Checkpoints** → [**Google Drive**](https://drive.google.com/drive/folders/19EANSjwyIp-z36jwCvKhwlWdcwA1Ohhq)
> (Anyone with the link → Viewer)

Two independently pretrained decoder-only Transformers, identical in shape,
sharing no data, tokenizer, vocabulary, or weights.

| | Model H (Hindi) | Model L (Nepali) |
| :--- | ---: | ---: |
| Trainable parameters | **25,379,840** | **25,379,840** |
| Training tokens (1 epoch) | 468,638,315 | 571,113,472 |
| Steps × tokens/step | 14,329 × 32,768 | 17,429 × 32,768 |
| Wall-clock (T4, fp16) | 3.69 h | 4.87 h |
| Throughput | 36,898 tok/s | 36,457 tok/s |
| Final train loss | 3.4603 | 3.5265 |

Held-out metrics, measured after training over the **complete** validation and
test splits (`src/eval/intrinsic.py`, 58M tokens scored, every token predicted
exactly once):

| | Model H (Hindi) | Model L (Nepali) |
| :--- | ---: | ---: |
| Test cross-entropy | 3.2885 | 3.4228 |
| **Test perplexity** | **26.80** | 30.65 |
| **Test bits-per-byte** | 0.5371 | **0.4495** |
| Test bits-per-character | 1.3636 | **1.1964** |
| Validation perplexity | 26.77 | 30.64 |
| Bytes / token | 8.833 | 10.985 |
| Characters / token | 3.479 | 4.127 |

**These are not the numbers the training curve shows, and the difference is
itself a finding.** During training the loop validated on `val_tokens:
2000000` — a fixed 2M-token *prefix* of the validation split. The Phase 1 splits
are written grouped by source, so that prefix contains Wikibooks, Wikiquote, the
news crawl and part of the manual web crawl, and never reaches `sangraha`, which
is **71.6%** of Hindi validation documents and **77.5%** of Nepali's. The prefix
is both unrepresentative and easier, and it reports perplexity about 30% below
the truth (H: 20.19 → 26.77; L: 23.77 → 30.64).

Checkpoint selection is unaffected: validation loss was *strictly* monotonic
across all 29 and 35 evaluations, so the best checkpoint is the last one under
any consistent measure. But the curve must be read as "a consistent progress
signal on one slice", not "the model's perplexity", and
`report/figures/val_perplexity.png` plots both so the gap is visible rather than
buried. Every number quoted in the report is the full-split one.

**The three metrics disagree, and that is the finding.** Hindi wins on perplexity
by 14.4%; Nepali wins on bits-per-byte by 19.5% and bits-per-character by 14.0%.
The cause is tokenizer fertility: Nepali's 8k BPE packs **4.127 chars/token
against Hindi's 3.479** (+18.6%), so a Nepali token is a harder prediction that
covers more text. Per-token perplexity penalises this; per-character and per-byte
normalisation reward it. This is precisely why the brief requires BPB *"to make
the two languages more comparable when tokenizers differ"* — and why the naive
reading, *"Hindi is the better model because its perplexity is lower"*, is wrong.

BPB is not fully neutral either: Nepali averages **10.985 bytes/token** against
Hindi's **8.833**, so BPB is deflated for Nepali by UTF-8 encoding density rather
than by model quality. Bits-per-character is the fairest of the three and still
favours Nepali, by a smaller margin. Report §2 works through all three.

**Neither model overfitted.** Best validation loss landed on the *final* step in
both runs, across 29 and 35 evaluations, with monotonically decreasing validation
loss throughout. The full-split measurements confirm it independently: validation
and test agree to three significant figures for both models (H: 26.77 / 26.80;
L: 30.64 / 30.65), which is what a model that has not memorised its validation
set looks like. Both runs are data-limited rather than converged — they were
still improving when the token budget was exhausted.

**Corpus sizes are measured, not estimated.** Every byte of all six split files
was tokenized with the committed Phase 1 models on 2026-09-02: Hindi
**520,392,597** tokens and Nepali **632,746,902**. These confirm the Phase 1
estimates above to within **0.006%** — Phase 1 derived its counts as
characters ÷ measured chars-per-token, and that approximation holds.

> [!NOTE]
> **Model H and Model L have identical parameter counts** because both Phase 1
> vocabularies came out at exactly 8,000. That makes the H-vs-L comparison a
> controlled test of the *data* rather than of capacity, and it is why the
> Transformer implementation lives once in `src/` instead of being duplicated
> per language.

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

> [!NOTE]
> **Phase 2 layout: shared implementation, per-language artifacts.**
> Guideline 7 of the brief asks for the repository organized by language; guideline 1
> asks for modular, non-duplicated code. Both Phase 1 vocabularies came out at exactly
> **8,000**, so Model H and Model L are *architecturally identical* — same 25,379,840
> parameters, same tensor shapes. Duplicating `model/`, `train/`, and `eval/` into two
> trees would mean maintaining two copies of the same network, where a fix applied to
> one and missed in the other silently breaks the very H-vs-L comparison the project
> is built on.
>
> So the Transformer implementation lives once under `src/`, while **`hindi/` and
> `nepali/` are self-contained for everything that actually differs**: configs, training
> logs, loss curves, evaluation outputs, generated samples, attention figures, and
> checkpoint links. The constraint guideline 7 protects is fully met — **the two models
> share no data, tokenizer, vocabulary, or weights.** Phase 1 already set this precedent
> with a shared `src/data/` alongside `data/{lang}` and `tokenizers/{lang}`.

```text
hindi/                               # Model H — self-contained artifacts
├── README.md                        #   corpus counts, run command, Drive links
├── configs/model.yaml               #   architecture (25,379,840 params)
├── configs/train.yaml               #   optimizer, schedule, paths
├── logs/train.jsonl                 #   per-step log -> loss curve
├── figures/ eval/ samples/          #   plots, metrics, generations
nepali/                              # Model L — same structure
src/
├── data/pack_tokens.py              # JSONL -> uint16 token stream
├── model/                           # embeddings, attention, block, GPT
├── train/                           # dataset, trainer, checkpointing
└── eval/                            # PPL/BPB, generation, metrics, attention
tests/                               # causal mask, param count, checkpoint resume
```

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
│   ├── remaining_hindi_manual_collection.ipynb  # Hindi news crawl -> manual_news_shard_0033-0038
│   ├── ocr_nepali_books_colab.ipynb # OCR route for Nepali print sources
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
│       └── train_bpe.py             # SentencePiece BPE trainer (configurable vocab)
├── requirements.txt                 # Pinned collection / tokenizer dependencies
└── README.md
```

---

## The Engineering Journey

*Lessons 1–14 are Phase 1 (collection, cleaning, tokenizer); 15–19 are Phase 2
(packing, training, checkpointing). Each records a decision that cost something
to learn.*

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

### Phase 2 — Packing, Training, and Checkpointing

### 15. Pack the Corpus Once, Not Once Per Step

* **The Challenge**: Tokenizing on the fly inside the training loop would have put
  SentencePiece encoding and Drive I/O on the critical path of all 31,758 steps.
* **The Solution**: `src/data/pack_tokens.py` encodes each split once into a flat
  `uint16` array. The vocabulary is 8,000, so a token id fits in two bytes:
  Hindi's training text drops **3.86 GiB → 0.87 GiB** and Nepali's
  **5.83 GiB → 1.06 GiB**. The trainer memory-maps the result, so a batch is a
  slice rather than a parse. This is a lossless change of representation — the
  packed output was verified byte-identical to a sequential single-process
  encode — not a subsample.

### 16. `\n\n` Is Not a Document Boundary

* **The Challenge**: `create_splits.py` writes the plain-text splits as
  `text.strip() + "\n\n"`, so the obvious way to place `<eos>` is to split on a
  blank line. But **81.9% of documents contain internal blank lines**. Splitting
  Hindi's `train.txt` that way yields **1,859,604 blocks against 848,815 real
  documents — 2.19×**.
* **Why it matters**: `<eos>` at 2.2× too many positions teaches the model that
  documents end mid-article, which is exactly what makes generation terminate
  early.
* **The Solution**: Pack from the `.jsonl`, one JSON object per line. The
  resulting `<eos>` counts came to **848,815** and **1,367,965** — matching the
  Phase 1 document counts exactly, to the document.

### 17. Bits-Per-Byte Must Count Text Bytes, Not File Bytes

* **The Challenge**: The packer's first version counted `len(raw)` per JSONL line
  for its `n_bytes_utf8` statistic. That line also carries `doc_id`, `source`,
  `collection_type`, `language`, and `text_sha256` — **45% envelope on a typical
  record**.
* **Why it matters**: `BPB = CE / (ln2 × bytes_per_token)`. Inflating
  bytes/token by 6% **understates BPB by 5.7%**, flattering both models, and the
  envelope ratio differs between the two languages — so it would have skewed the
  H-vs-L comparison in an unpredictable direction.
* **How it surfaced**: the packer reported 9.38 bytes/token for Hindi where an
  earlier independent pass over the `.txt` files had measured **8.85**. One
  number disagreeing with another is the only reason it was caught.
* **The Solution**: count `len(text.encode("utf-8"))`, and a `--repair-bytes`
  mode that corrects existing sidecars without re-tokenizing.

### 18. A Resume Test That Passes on CPU Proves Nothing About the GPU

* **The Challenge**: `test_checkpoint_resume.py` asserted that training after a
  save/load cycle reproduces uninterrupted training *exactly* — and it passed.
  The first real GPU resume then died with
  `TypeError: RNG state must be a torch.ByteTensor`.
* **The Cause**: the trainer loaded with `map_location="cuda"`, which moves
  **every** tensor in the payload to the GPU, RNG states included.
  `set_rng_state_all()` requires CPU byte tensors. The tests saved and loaded on
  CPU, so the only code path that ever runs in production was never exercised.
* **The Solution**: load with `map_location="cpu"` (`load_state_dict` copies into
  an already-on-GPU model regardless), coerce RNG states back to CPU `uint8`
  anyway, and make restoration **best-effort** — correct weights matter, an
  exactly replayed dropout stream does not. Plus two regression tests that
  simulate a GPU-shaped payload.

### 19. A Smoke Run Must Not Be Able to Touch Real Checkpoints

* **The Challenge**: the 100-step smoke run wrote to a throwaway `--ckpt-dir`,
  but `mirror_to` was still active in the config, so its toy checkpoints were
  pushed to the real Drive folder. The subsequent full run found them and
  resumed a 4-hour job from a 100-step model.
* **The Solution**: a `--no-mirror` flag, used by both notebooks' smoke cells, and
  the contaminated remote purged before relaunching.

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
A document's split is a function of its content hash, so assignment is stable
regardless of shard ingest order and no document can leak across splits.

| Split | Purpose | Hindi tokens | Nepali tokens |
| :--- | :--- | ---: | ---: |
| **Train** (90%) | Pretraining, and tokenizer fitting | 467,905,255 | 569,663,061 |
| **Validation** (5%) | Loss, perplexity (PPL), bits-per-byte (BPB) | 26,069,560 | 31,697,868 |
| **Test** (5%) | Generation quality (BLEU, chrF, ROUGE-L), attention analysis | 26,434,593 | 31,350,379 |
| **Total** | | **520,409,408** | **632,711,308** |

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
* **Manual Data (118 Shards — 2.217 GiB, post-deduplication)**: **307,528 documents
  / 549,501,090 unique clean characters / ~110.0M tokens** — the Nepali Wikipedia
  dump (27,946 docs, 100% unique), a sitemap-driven crawl over 22 Nepali portals
  (202,769 docs / 340.2M chars, collected 2026-08-21) and the legacy news crawl
  (*Ratopati*, *Setopati*, *Gorkhapatra*, *OnlineKhabar*), of which only **37.2%**
  survived content-hash deduplication.
* **Automated Data Shards (248 Shards — 6.263 GiB)**: ~544.3M tokens ingested from
  AI4Bharat Sangraha Verified, budget-limited from a much larger pool.
* **Total Cleaned Storage on Google Drive**: **366 shards (8.48 GiB)**.
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
* **Vocabulary Size**: **8,000** for both models, in two separate vocabularies. Selected on the measured parameter-efficiency rule in report §6.2.4; the course Q&A also advises 5–10K.
* **Character Coverage**: `0.9995` (retains Devanagari combining marks, nuktas, virama `्`).
* **Special Tokens**: `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`.
* **Script Preservation**: `split_by_unicode_script=True`, `split_by_whitespace=True`, `split_digits=True`.

### Measured on held-out text

Each model measured on its own validation split, never seen in training —
Hindi on 59,713 documents (12.2M characters), Nepali on 70,196 documents
(11.6M characters):

| Statistic | `hindi_bpe_8k` | `nepali_bpe_8k` |
| :--- | ---: | ---: |
| Fertility (tokens/word) | **1.4938** | **1.5566** |
| Characters per token | 3.4712 | 4.2537 |
| Unknown-token rate | **0.00%** | **0.00%** |
| Vocabulary coverage | 97.41% (7,793 of 8,000) | 96.99% (7,759 of 8,000) |
| Broken grapheme rate | 5.5683% | 4.462% |

Nepali compresses better at equal vocabulary — its bound case markers and plural
`-हरू` give BPE longer, more reusable stems (report §6.4).

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

### Why 8,000

Vocabulary size trades directly against the ~25M parameter budget: with
`d_model = 256` and weight tying, the embedding table costs `V x 256` parameters
— 8.2% of the budget at 8k, but **32.8% at 32k**. A vocabulary sweep on held-out
text (report §6.2) shows 8k -> 32k buys only 0.23 tokens/word of fertility while
vocabulary coverage collapses from 94.31% to 82.39%, leaving ~1.4M embedding
parameters that are never used. The course Q&A also advises a 5–10K vocabulary.

The same rule selects 8,000 for Nepali, though the argument differs: Nepali
coverage stays healthy at 32k (93.93%, against Hindi's 82.39%), so 32k is not
*wasteful* there, only unaffordable — at `d_model = 512` its embedding table alone
would be 65.5% of a 25M model. Nepali was originally trained at 32,000 and
retrained at 8,000 on 2026-08-21; the superseded model is kept on Drive and the
cost of the change is quantified in report §6.4 and §6.6 rather than smoothed over.

Full benchmarks, including the vocabulary sweep and the BPE-vs-Unigram comparison
at equal data: [`report/hindi_tokenizer_benchmark.json`](report/hindi_tokenizer_benchmark.json) ·
[`report/nepali_tokenizer_benchmark.json`](report/nepali_tokenizer_benchmark.json) ·
[`report/hindi_algorithm_comparison.json`](report/hindi_algorithm_comparison.json) ·
[`report/nepali_algorithm_comparison.json`](report/nepali_algorithm_comparison.json)
(reproduce with `scripts/benchmark_tokenizer.py`).

---

## Model Architecture and Pretraining

Built from primitive PyTorch layers only — `nn.Linear`, `nn.Embedding`,
`nn.LayerNorm`, `nn.Dropout`. No `nn.Transformer*`, no HuggingFace model classes,
no pre-built attention. Implementation in [`src/model/`](src/model); configs in
[`hindi/configs/model.yaml`](hindi/configs/model.yaml) and
[`nepali/configs/model.yaml`](nepali/configs/model.yaml).

| Hyperparameter | Value | Why |
| :--- | ---: | :--- |
| `d_model` | 512 | 8 heads × 64-dim head, the standard small-LM width |
| `n_layers` | 8 | Depth preferred over FFN width at a fixed budget |
| `n_heads` | 8 | `d_k = d_model / n_heads = 64` |
| `d_ff` | 1536 | 3× `d_model`. See the depth/width note below |
| `ctx_len` | 512 | Learned absolute positions cap sequences at exactly this |
| `dropout` | 0.1 | Embeddings, attention probabilities, sublayer outputs |
| `vocab_size` | 8,000 | Phase 1 BPE, one per language, never shared |
| Weight tying | yes | LM head shares the token embedding: saves 4,096,000 params |
| Positional encoding | learned absolute | Simple at fixed length, and trivially ablatable |
| Normalisation | pre-norm LayerNorm | Stable gradients through 8 residual blocks |
| Activation | GELU | GPT-2 standard |

### Exact parameter count — 25,379,840

| Component | Formula | Parameters |
| :--- | :--- | ---: |
| Token embedding | `8000 × 512` | 4,096,000 |
| Positional embedding | `512 × 512` | 262,144 |
| Per block — attention (Q,K,V,O + biases) | `4 × (512×512 + 512)` | 1,050,624 |
| Per block — FFN (+ biases) | `(512×1536+1536) + (1536×512+512)` | 1,574,912 |
| Per block — 2 LayerNorms | `2 × 2 × 512` | 2,048 |
| **Per block** | | **2,627,584** |
| × 8 blocks | | 21,020,672 |
| Final LayerNorm | `2 × 512` | 1,024 |
| LM head | tied — no extra | 0 |
| **Total** | | **25,379,840** |

`TransformerLMConfig.expected_params()` computes this in closed form and
[`tests/test_model_shapes.py`](tests/test_model_shapes.py) asserts the built model
matches it, so a wiring mistake cannot pass silently.

### Why `d_ff = 1536` and not 2048

The conventional 4× expansion (`d_ff = 2048`) yields **29.6M** parameters, well
over the ~25M target. Two ways to come back: drop to 6 layers at 4× (23.3M), or
keep 8 layers at 3× (25.4M). We kept the depth. For a model this small, additional
blocks buy compositional depth — more opportunities to re-attend and re-combine —
whereas a wider FFN only enlarges a per-position transform that is already
generously sized. A 2.67–3× expansion is also standard practice in SwiGLU-era
models, so 3× is well precedented rather than an arbitrary trim.

> [!NOTE]
> **Reconciling the Phase 1 vocabulary argument.** The *Why 8,000* section below
> reasons about the embedding budget assuming `d_model = 256`, where an 8k table
> costs 8.1% of the parameter budget. The shipped model uses `d_model = 512`, so
> the real figure is **16.1%** (4,096,000 of 25,379,840) — halved back to an
> effective 8.1% by weight tying, which reuses that same matrix as the output
> head. The Phase 1 conclusion is unchanged and the direction of the argument
> holds: at 32k the tied table alone would be 65.5% of the budget.

### Training configuration

| Setting | Value | Why |
| :--- | :--- | :--- |
| Optimizer | AdamW, `betas=(0.9, 0.95)` | `beta2=0.95` suits short runs |
| Weight decay | 0.1, matmul weights only | Biases and LayerNorm gains are 1-D and get none |
| Peak LR | 3e-4 | Standard for ~25M at this batch size |
| Schedule | 500-step linear warmup → cosine to 3e-5 | Adam's second moment is unreliable early; full-size steps then can land the model somewhere it never recovers from |
| Tokens/step | 32,768 | Hindi 16×512×4, Nepali 32×512×2 |
| Precision | fp16 AMP + GradScaler | The T4 has no bf16 |
| Gradient clipping | 1.0 | Observed grad norms settled at 0.7–1.0 |
| Epochs | 1 | ≈500M tokens, ~18.5 and 22.5 tokens/parameter |

**On the token budget.** Both runs land near the Chinchilla-optimal ratio of ~20
tokens per parameter (Hindi 18.5, Nepali 22.5), so the models are neither
undertrained nor overtrained for their size. A second epoch was considered and
rejected: extrapolating the late-run improvement rate gives roughly ppl
26.77 → roughly 23 for Hindi, the learning rate has already annealed to its floor
(so a second epoch means retraining from scratch at 2× cost, not continuing),
and the brief attaches no marks to perplexity. The same GPU hours are worth more
spent on the no-positional-embedding ablation.

**On batch size and gradient accumulation.** Both models see 32,768 tokens per
optimisation step, but reach it differently: Hindi at `16 × 4` and Nepali at
`32 × 2`. The gradient is mathematically identical either way. Batch 64 in a
single pass runs out of memory on a 15 GB T4 — cross-entropy upcasts logits to
fp32, making `64 × 512 × 8000 × 4 B = 1000 MiB` in one tensor, and the explicit
`(B, h, T, T)` attention matrices scale with batch size too. Hindi ran first and
peaked at 3.9 GiB, which showed batch 32 was safe for Nepali.

### Checkpointing

Mandatory per the brief: *"A run that cannot resume after interruption will lose
marks."* Every checkpoint in [`src/train/checkpoint.py`](src/train/checkpoint.py)
carries the five required fields — model weights, optimizer state, scheduler
state, training step, configuration — plus four more that an interrupted run
actually needs: **GradScaler state** (its loss scale adapts during training),
**RNG states**, `best_val_loss`, and `tokens_seen`.

Writes are **atomic** (`.tmp` → `fsync` → `os.replace`). A session killed midway
through a 305 MB write to a network mount would otherwise leave a truncated file
exactly where the resume path expects a valid one, converting a recoverable
disconnect into a lost run. `find_resume_point()` falls back to the newest intact
milestone if `latest` is unreadable.

| | Cadence | Size |
| :--- | :--- | ---: |
| `checkpoint_latest.pt` | every 500 steps | 304.7 MB |
| `checkpoint_best.pt` | on validation improvement | 304.7 MB |
| `checkpoint_step_N.pt` | every 2,000 steps, 6 kept | 304.7 MB |
| `config.json` | once, beside the weights | — |

304.7 MB is fp32 weights (101.5 MB) plus AdamW's two moment buffers. At a
500-step cadence a crash costs at most ~3 minutes of compute.

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
   * Train the 8k SentencePiece BPE tokenizer and save all artifacts to Drive.

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
    --model-prefix nepali/tokenizer/nepali_bpe_8k \
    --vocab-size 8000 \
    --language nepali
```

---

## Reproducing Phase 2

All Phase 2 work runs on free Colab / Kaggle GPUs. Notebooks are thin
orchestration; the logic lives in `src/`.

### 1. Pack the corpus (CPU, ~30–50 min)

[`notebooks/phase2_setup_colab.ipynb`](notebooks/phase2_setup_colab.ipynb) —
mounts Drive, encodes the JSONL splits into `uint16` streams, verifies the token
counts, runs the test suite, and prints the parameter breakdown.

```bash
python -m src.data.pack_tokens \
    --input-dir  <drive>/hindi/splits \
    --tokenizer  tokenizers/hindi/hindi_bpe_8k.model \
    --output-dir <drive>/hindi/tokens \
    --language   hindi
```

Verification asserts the packed counts against figures measured independently by
streaming and tokenizing every byte of all six split files:

| | train | val | test | total |
| :--- | ---: | ---: | ---: | ---: |
| Hindi | 467,874,360 | 26,127,194 | 26,391,043 | **520,392,597** |
| Nepali | 569,684,621 | 31,676,467 | 31,385,814 | **632,746,902** |

### 2. Tests (~15 s)

```bash
python -m pytest tests/ -v
```

23 tests. The load-bearing ones: `test_causal_mask.py` verifies empirically that
changing the token at `t+1` leaves the logits at every position `≤ t` bit-for-bit
unchanged (required by §2.1 — a broken mask trains normally and reports an
excellent validation loss while copying from the future);
`test_checkpoint_resume.py` asserts that resumed training produces *identical*
losses to uninterrupted training, not merely that weights match.

### 3. Pretrain

```bash
python -m src.train.trainer --config hindi/configs/train.yaml     # Colab,  3.69 h
python -m src.train.trainer --config nepali/configs/train.yaml    # Kaggle, 4.87 h
```

[`notebooks/pretrain_hindi_colab.ipynb`](notebooks/pretrain_hindi_colab.ipynb) ·
[`notebooks/pretrain_nepali_kaggle.ipynb`](notebooks/pretrain_nepali_kaggle.ipynb)

Both notebooks run a 100-step smoke test first — it catches a bad config or an
OOM in about a minute rather than hours into a run, and it is what caught the
batch-64 out-of-memory failure described above.

**Resume needs no flag.** The trainer looks for `checkpoint_latest.pt` in the
configured directory and continues from it, restoring model, optimizer,
scheduler, GradScaler, step, and RNG state. Re-running the same command after a
disconnect is the entire recovery procedure.

Kaggle has no `drive.mount`, so the Nepali notebook uses **rclone** instead: it
pulls the packed tokens in at session start and mirrors every checkpoint back out
to Drive. That second half is not optional — `/kaggle/working` is discarded when
an interactive session dies, so an unmirrored run would lose everything on a
disconnect.

### 4. Logs and figures

Training logs are written to persistent storage during the run, then pulled into
the repository, because *"if it is not in the branch, it is not graded"*:
[`hindi/logs/train.jsonl`](hindi/logs/train.jsonl) ·
[`nepali/logs/train.jsonl`](nepali/logs/train.jsonl)

---

## Google Drive Artifacts and Checkpoints

All large artifacts live in a single shared Google Drive folder:

## [**`24KaratAu/` — all datasets, splits and tokenizers**](https://drive.google.com/drive/folders/19EANSjwyIp-z36jwCvKhwlWdcwA1Ohhq)

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
| Cleaned dataset — manual (JSONL) | `ne` | `nepali/cleaned/manual/` | 2.217 GiB |
| Raw crawled HTML (gzipped per shard) | `ne` | `nepali/raw/manual_web/` | 2.193 GiB |
| Raw Wikimedia dump (`.xml.bz2`) | `ne` | `nepali/raw/wiki_dump/` | 0.051 GiB |
| Train / val / test splits | `ne` | `nepali/splits/` | 13.348 GiB |
| BPE tokenizer model + vocab (`8k`) | `ne` | `nepali/tokenizer/nepali_bpe_8k.{model,vocab}` | 599 KiB |
| Superseded tokenizer (`32k` BPE + Unigram) | `ne` | `nepali/tokenizer/` | 4 MiB |
| Packed token streams (`uint16`) | `hi` | `hindi/tokens/hindi_{train,val,test}.bin` | 0.97 GiB |
| Packed token streams (`uint16`) | `ne` | `nepali/tokens/nepali_{train,val,test}.bin` | 1.18 GiB |
| **Pretrained checkpoint — Model H** | `hi` | `hindi/checkpoints/checkpoint_best.pt` | 304.7 MB |
| Model H milestones + latest | `hi` | `hindi/checkpoints/` | 2.4 GiB |
| **Pretrained checkpoint — Model L** | `ne` | `nepali/checkpoints/checkpoint_best.pt` | 304.7 MB |
| Model L milestones + latest | `ne` | `nepali/checkpoints/` | 3.0 GiB |
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
