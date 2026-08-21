# Data Collection Ledger

**Generated from `report/collection_ledger.json` by `scripts/render_collection_ledger.py`** — edit the JSON, not this file.

Definitive record of what was collected, by which script or notebook, at which commit, and into which shards. Times are IST (Drive stores UTC; +05:30 applied). Document and character counts are measured unless marked 'estimated', which means extrapolated from a sampled subset by byte size.

`where` is **colab** for a notebook cell run on a Google Colab runtime writing
to the mounted Drive, and **local** for a script run on the author's machine
streaming to Drive over `rclone`.

---

## At a glance

| # | Date (IST) | Where | Language | Ran | Shards produced |
| ---: | :--- | :--- | :--- | :--- | :--- |
| 1 | 2026-08-13 -> 08-14 | local | nepali | `src/data/collect/collect_manual_web.py, collect_wikimedia_a…` | data/nepali/raw/manual_web/ (local HTML), nepali/corpus… |
| 2 | 2026-08-15 18:26 -> 21:58 | colab | nepali | `notebooks/collect_nepali_colab.ipynb (iterated)` | nepali/raw/wiki_dump/newiki-latest-pages-articles.xml.b… |
| 3 | 2026-08-15 21:58 -> 08-16 02:09 | colab | nepali | `collect_nepali_colab.ipynb — Sangraha ingest` | nepali/cleaned/automated_nonweb/nonweb_shard_*.jsonl |
| 4 | 2026-08-16 02:09 -> 02:10 | colab | nepali | `collect_nepali_colab.ipynb — Wikipedia dump parse` | nepali/cleaned/manual/wiki_shard_0000-0005.jsonl |
| 5 | 2026-08-16 02:17 -> 03:02 | colab | nepali | `collect_nepali_colab.ipynb — tokenizer training` | nepali/tokenizer/nepali_bpe_32k.{model,vocab}, nepali_u… |
| 6 | 2026-08-17 20:00 -> 23:27 | colab | nepali | `collect_nepali_colab.ipynb — news crawl` | nepali/cleaned/manual/manual_news_shard_0000-0041.jsonl |
| 7 | 2026-08-17 23:31 | colab | nepali | `collect_nepali_colab.ipynb — split` | nepali/splits/{train,val,test}.{txt,jsonl} |
| 8 | 2026-08-19 01:19 -> 01:26 | colab | hindi | `notebooks/collect_hindi_colab.ipynb — Step 2a` | hindi/cleaned/manual/wiki_shard_0000-0029.jsonl |
| 9 | 2026-08-19 01:26 -> 01:59 | colab | hindi | `collect_hindi_colab.ipynb — Step 1 Sangraha ingest` | hindi/cleaned/automated_nonweb/nonweb_shard_*.jsonl |
| 10 | 2026-08-19 14:46 | colab | hindi | `collect_hindi_colab.ipynb — literature step` | hindi/cleaned/manual/manual_lit_shard_0030.jsonl |
| 11 | 2026-08-20 19:17 | colab | hindi | `collect_hindi_colab.ipynb — Step 2c` | hindi/cleaned/manual/manual_aux_shard_0032.jsonl |
| 12 | 2026-08-20 19:56 -> 22:50 | colab | hindi | `notebooks/remaining_hindi_manual_collection.ipynb (v3, supe…` | hindi/cleaned/manual/manual_news_shard_0033-0038.jsonl |
| 13 | 2026-08-20 20:06 | colab | hindi | `collect_hindi_colab.ipynb — Wikisource step` | hindi/cleaned/manual/wikisource_shard_0000.jsonl |
| 14 | 2026-08-20 22:36 -> 23:07 | local | hindi | `scripts/discover_sitemap_urls.py; scripts/export_manual_has…` | hindi/manifests/frontier_full.txt.gz, frontier_balanced… |
| 15 | 2026-08-20 23:13 -> 08-21 00:27 | local | hindi | `scripts/collect_manual_web_sitemap.py --language hindi` | hindi/cleaned/manual/manual_web_shard_0000-0031.jsonl +… |
| 16 | 2026-08-21 03:15 -> 04:09 | colab | hindi | `notebooks/finalize_hindi_corpus.ipynb` | hindi/splits/{train,val,test}.{txt,jsonl}; hindi/tokeni… |
| 17 | 2026-08-21 04:32 -> 04:48 | local | hindi | `curl + rclone (report 7.5)` | hindi/raw/wiki_dump/{hiwiki,hiwikibooks,hiwikiquote}-la… |
| 18 | 2026-08-21 05:17 -> 05:55 | local | nepali | `exact val.txt measurement; scripts/export_manual_hashes.py;…` | data/nepali/sources/frontier.txt, frontier_balanced.txt… |
| 19 | 2026-08-21 05:40 -> 09:36 | local | nepali | `scripts/collect_manual_web_sitemap.py --language nepali` | nepali/cleaned/manual/manual_web_shard_*.jsonl + nepali… |
| 20 | 2026-08-21 12:45 -> 13:30 | colab | nepali | `notebooks/finalize_nepali_corpus.ipynb` | nepali/splits/{train,val,test}.{txt,jsonl}; nepali/toke… |

---

## Stage detail

### Stage 1 — Nepali, 2026-08-13 -> 08-14 (local)

* **Commits**: `552605f`, `b263dd2`, `1ed2dd5`, `f5e4972`, `a7e3dfc`
* **Ran**: `src/data/collect/collect_manual_web.py, collect_wikimedia_api.py`
* **Source**: Nepali news + Wikipedia via API; pilot/test runs
* **Produced**: `data/nepali/raw/manual_web/ (local HTML), nepali/corpus/, nepali/raw/test/ on Drive`

> Development and pilot phase. The raw/test and corpus/test folders on Drive are fixtures from scripts/test_shard_rollover.py and test_upload_failure_recovery.py.

---

### Stage 2 — Nepali, 2026-08-15 18:26 -> 21:58 (colab)

* **Commits**: `eca89ce`, `146874e`, `897638d`, `e9e5aeb`, `d63d026`, `85d7550`, `351858c`, `69fd95c`, `082ac21`, `73fcf5f`, `db8f608`
* **Ran**: `notebooks/collect_nepali_colab.ipynb (iterated)`
* **Source**: debugging: HF login, XML namespace, ElementTree RAM leak, data-0.parquet, PyArrow micro-batching
* **Produced**: `nepali/raw/wiki_dump/newiki-latest-pages-articles.xml.bz2 (54,964,181 B)`

---

### Stage 3 — Nepali, 2026-08-15 21:58 -> 08-16 02:09 (colab)

* **Commits**: `db8f608`
* **Ran**: `collect_nepali_colab.ipynb — Sangraha ingest`
* **Source**: HuggingFace ai4bharat/sangraha (verified/nep)
* **Produced**: `nepali/cleaned/automated_nonweb/nonweb_shard_*.jsonl`
* **Counts toward**: **automated**
* **Volume**: 248 shards

| Measurement | Value |
| :--- | ---: |
| documents | ~1,090,000 (estimated) |
| characters | 2,445,256,140 |
| chars per token | 4.493 |
| tokens | 544,293,393 |

> Basis: 3 shards sampled, scaled by byte size

---

### Stage 4 — Nepali, 2026-08-16 02:09 -> 02:10 (colab)

* **Commits**: `db8f608`
* **Ran**: `collect_nepali_colab.ipynb — Wikipedia dump parse`
* **Source**: newiki-latest-pages-articles.xml.bz2
* **Produced**: `nepali/cleaned/manual/wiki_shard_0000-0005.jsonl`
* **Counts toward**: **manual**
* **Volume**: 6 shards

| Measurement | Value |
| :--- | ---: |
| documents | 27,946 |
| characters | 42,864,542 |
| unique documents | 27,946 |
| unique characters | 42,864,542 |
| uniqueness | 100% |

---

### Stage 5 — Nepali, 2026-08-16 02:17 -> 03:02 (colab)

* **Commits**: `db8f608`
* **Ran**: `collect_nepali_colab.ipynb — tokenizer training`
* **Source**: nepali splits (superseded)
* **Produced**: `nepali/tokenizer/nepali_bpe_32k.{model,vocab}, nepali_unigram_32k.{model,vocab}`

| Measurement | Value |
| :--- | ---: |
| fertility | 1.317 |
| chars per token | 5.0274 |
| coverage pct | 91.92 |
| unknown rate pct | 0.0 |
| broken grapheme pct | 1.609 |

> Basis: 70,196 held-out validation documents, measured 2026-08-21

---

### Stage 6 — Nepali, 2026-08-17 20:00 -> 23:27 (colab)

* **Commits**: `db8f608`
* **Ran**: `collect_nepali_colab.ipynb — news crawl`
* **Source**: Ratopati, Setopati, Gorkhapatra, OnlineKhabar
* **Produced**: `nepali/cleaned/manual/manual_news_shard_0000-0041.jsonl`
* **Counts toward**: **manual**
* **Volume**: 42 shards

| Measurement | Value |
| :--- | ---: |
| documents | 208,975 |
| characters | 447,189,127 |
| unique documents | 76,813 |
| unique characters | 166,421,339 |
| uniqueness | 37.2% of characters |

> **URL-deduplicated only; syndicated copy repeated under disjoint URLs (report 4.3)**

---

### Stage 7 — Nepali, 2026-08-17 23:31 (colab)

* **Commits**: `db8f608`
* **Ran**: `collect_nepali_colab.ipynb — split`
* **Produced**: `nepali/splits/{train,val,test}.{txt,jsonl}`

| Measurement | Value |
| :--- | ---: |
| characters | 516,682,122 |
| tokens | ~102,700,000 |
| val txt exact characters | 25,500,994 |

> **SUPERSEDED: captured ~16% of the available pool; no ingest assertion**

---

### Stage 8 — Hindi, 2026-08-19 01:19 -> 01:26 (colab)

* **Commits**: `add5f40`, `973bf4d`
* **Ran**: `notebooks/collect_hindi_colab.ipynb — Step 2a`
* **Source**: dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2 (239,196,553 B)
* **Produced**: `hindi/cleaned/manual/wiki_shard_0000-0029.jsonl`
* **Counts toward**: **manual**
* **Volume**: 30 shards

| Measurement | Value |
| :--- | ---: |
| unique documents | 146,296 |
| unique characters | 253,116,189 |

---

### Stage 9 — Hindi, 2026-08-19 01:26 -> 01:59 (colab)

* **Commits**: `add5f40`
* **Ran**: `collect_hindi_colab.ipynb — Step 1 Sangraha ingest`
* **Source**: HuggingFace ai4bharat/sangraha (verified/hin)
* **Produced**: `hindi/cleaned/automated_nonweb/nonweb_shard_*.jsonl`
* **Counts toward**: **automated**
* **Volume**: 220 shards

| Measurement | Value |
| :--- | ---: |
| documents | ~1,090,000 (estimated) |
| characters | ~2,400,700,000 (estimated) |
| chars per token | 3.507 |
| tokens | ~684,500,000 (estimated) |

> Basis: 8 shards sampled, scaled by byte size

---

### Stage 10 — Hindi, 2026-08-19 14:46 (colab)

* **Commits**: `696c060`, `6b8c6f4`
* **Ran**: `collect_hindi_colab.ipynb — literature step`
* **Source**: re-parse of the same Wikimedia dumps as stage 12
* **Produced**: `hindi/cleaned/manual/manual_lit_shard_0030.jsonl`
* **Counts toward**: **manual**
* **Volume**: 1 shards

| Measurement | Value |
| :--- | ---: |
| documents | 2,366 |
| unique documents | 1 |

> **99.96% redundant: 2,365 of 2,366 documents already present in manual_aux_shard_0032**

---

### Stage 11 — Hindi, 2026-08-20 19:17 (colab)

* **Commits**: `baf4076`
* **Ran**: `collect_hindi_colab.ipynb — Step 2c`
* **Source**: hiwikibooks + hiwikiquote -latest-pages-articles.xml.bz2
* **Produced**: `hindi/cleaned/manual/manual_aux_shard_0032.jsonl`
* **Counts toward**: **manual**
* **Volume**: 1 shards

| Measurement | Value |
| :--- | ---: |
| documents | 4,058 |
| unique documents | 4,015 |
| unique characters | 24,368,410 |

---

### Stage 12 — Hindi, 2026-08-20 19:56 -> 22:50 (colab)

* **Commits**: `baf4076`
* **Ran**: `notebooks/remaining_hindi_manual_collection.ipynb (v3, superseded)`
* **Source**: Hindi news portals
* **Produced**: `hindi/cleaned/manual/manual_news_shard_0033-0038.jsonl`
* **Counts toward**: **manual**
* **Volume**: 6 shards

| Measurement | Value |
| :--- | ---: |
| unique documents | 14,811 |
| unique characters | 41,116,565 |

---

### Stage 13 — Hindi, 2026-08-20 20:06 (colab)

* **Commits**: `baf4076`
* **Ran**: `collect_hindi_colab.ipynb — Wikisource step`
* **Source**: hiwikisource-latest-pages-articles.xml.bz2
* **Produced**: `hindi/cleaned/manual/wikisource_shard_0000.jsonl`
* **Counts toward**: **manual**
* **Volume**: 1 shards

| Measurement | Value |
| :--- | ---: |
| documents | 1 |
| unique documents | 0 |

> **FAILED: filtered ns in {0,104}, but hi.wikisource uses namespace 250. Re-parsing with --namespaces 0,250 recovers 161,734 docs / 268,055,655 chars. Not ingested (report 3.3).**

---

### Stage 14 — Hindi, 2026-08-20 22:36 -> 23:07 (local)

* **Commits**: `baf4076`
* **Ran**: `scripts/discover_sitemap_urls.py; scripts/export_manual_hashes.py`
* **Source**: robots.txt Sitemap directives across 18 Hindi domains
* **Produced**: `hindi/manifests/frontier_full.txt.gz, frontier_balanced.txt.gz, existing_manual_hashes.txt.gz`

| Measurement | Value |
| :--- | ---: |
| urls discovered | 11,112,623 |
| balanced frontier urls | 375,539 |
| dedup registry hashes | 155,123 |
| discovery seconds | 456 |

---

### Stage 15 — Hindi, 2026-08-20 23:13 -> 08-21 00:27 (local)

* **Commits**: `baf4076`
* **Ran**: `scripts/collect_manual_web_sitemap.py --language hindi`
* **Source**: 32 Hindi news/government portals (indiatv 19.0%, livehindustan 17.3%, amarujala 16.8%, pardaphash 13.7%, moneycontrol 9.0%, patrika 8.5%, bhaskar 5.5%, thelallantop 4.4%, hi.vikaspedia 1.4%)
* **Produced**: `hindi/cleaned/manual/manual_web_shard_0000-0031.jsonl + hindi/raw/manual_web/*.tar.gz`
* **Counts toward**: **manual**
* **Volume**: 27 shards, 27 raw archives

| Measurement | Value |
| :--- | ---: |
| urls attempted | 87,585 |
| documents | 75,396 |
| characters | 161,000,865 |
| unique documents | 75,387 |
| unique characters | 160,980,352 |
| acceptance rate | 0.86 |
| rate url per s | 20.2 |
| chars per token | 3.538 |

---

### Stage 16 — Hindi, 2026-08-21 03:15 -> 04:09 (colab)

* **Commits**: `baf4076`
* **Ran**: `notebooks/finalize_hindi_corpus.ipynb`
* **Produced**: `hindi/splits/{train,val,test}.{txt,jsonl}; hindi/tokenizer/hindi_bpe_8k.{model,vocab}`

| Measurement | Value |
| :--- | ---: |
| documents | 848,815 |
| characters | 1,813,233,513 |
| tokens | 520,409,408 |
| manual share pct | 26.87 |
| chars per token | 3.484 |
| manual duplicates dropped | 2,523 |
| duplicate characters | 18,639,999 |
| train tokens | 467,905,255 |
| val tokens | 26,069,560 |
| test tokens | 26,434,593 |

---

### Stage 17 — Hindi, 2026-08-21 04:32 -> 04:48 (local)

* **Commits**: `e65804c`
* **Ran**: `curl + rclone (report 7.5)`
* **Source**: dumps.wikimedia.org
* **Produced**: `hindi/raw/wiki_dump/{hiwiki,hiwikibooks,hiwikiquote}-latest-pages-articles.xml.bz2`

| Measurement | Value |
| :--- | ---: |
| bytes | 251,074,420 |

> **Re-fetched 2026-08-21; Wikimedia regenerates 'latest' continuously, so these are the same sources at a later snapshot than the 19 Aug parse.**

---

### Stage 18 — Nepali, 2026-08-21 05:17 -> 05:55 (local)

* **Commits**: `e65804c`
* **Ran**: `exact val.txt measurement; scripts/export_manual_hashes.py; scripts/discover_sitemap_urls.py`
* **Source**: audit + robots.txt Sitemap discovery over 38 probed Nepali domains
* **Produced**: `data/nepali/sources/frontier.txt, frontier_balanced.txt; existing_manual_hashes.txt`

| Measurement | Value |
| :--- | ---: |
| urls discovered | 651,660 |
| balanced frontier urls | 604,392 |
| dedup registry hashes | 100,929 |
| manual uniqueness pct | 42.7 |

> **ekantipur, onlinekhabar, ratopati, himalkhabar, nagariknews, annapurnapost publish no sitemap**

---

### Stage 19 — Nepali, 2026-08-21 05:40 -> 09:36 (local)

* **Commits**: `e65804c`
* **Ran**: `scripts/collect_manual_web_sitemap.py --language nepali`
* **Source**: 22 Nepali portals (makalukhabar 22.4%, kathmandupati 20.7%, bizmandu 16.1%, clickmandu 15.6%, arthasarokar 14.0%, deshsanchar 5.9%, nepalnews 4.9%)
* **Produced**: `nepali/cleaned/manual/manual_web_shard_*.jsonl + nepali/raw/manual_web/*.tar.gz`
* **Counts toward**: **manual**
* **Volume**: 70 shards, 73 raw archives

| Measurement | Value |
| :--- | ---: |
| documents | 202,769 |
| characters | 340,215,209 |
| duplicates in 9000 doc sample | 0 |
| chars per token | 5.091 |
| rate url per s | 24.6 |
| mean chars per doc | 1,682 |
| rejections | http_404 11,448, langid_uncertain 5,883, quality_filter 2,543, http_429 1,516, duplicate 1,002, langid_reject_hi 17 |

---

### Stage 20 — Nepali, 2026-08-21 12:45 -> 13:30 (colab)

* **Commits**: `e65804c`
* **Ran**: `notebooks/finalize_nepali_corpus.ipynb`
* **Source**: all 118 manual shards + Sangraha, budget-limited
* **Produced**: `nepali/splits/{train,val,test}.{txt,jsonl}; nepali/tokenizer/nepali_bpe_8k.{model,vocab}`
* **Counts toward**: **mixed**

| Measurement | Value |
| :--- | ---: |
| manual documents | 303,275 |
| manual characters | 541,814,473 |
| manual duplicates dropped | 132,181 |
| duplicate characters | 280,791,280 |
| automated characters | 2,069,272,506 |
| total documents | 1,367,965 |
| total characters | 2,611,086,979 |
| total tokens | 632,711,308 |
| manual share pct | 21.2 |
| chars per token | 4.127 |
| train tokens | 569,663,061 |
| val tokens | 31,697,868 |
| test tokens | 31,350,379 |
| manual chars per token 8k | 4.014 |
| automated chars per token 8k | 4.123 |

> **FINAL. The automated budget was derived from chars/token = 5.022 measured on the pre-existing nepali_bpe_32k, but the corpus was then tokenised with the newly trained 8k model, so 2.61B characters produced 632.7M tokens rather than the projected 520M — over the ~500M target rather than under it. The manual share was re-measured directly against nepali_bpe_8k at 21.20% (manual 4.014 chars/token, automated 4.123); Cell 3's 20.75% used the 32k ratios and understated it. Both requirements are met, so no re-split was run. Separately, val.jsonl and test.jsonl initially failed to leave the FUSE mount while the sync reported success (report 8.1 failure 6) and were re-uploaded through the Drive API; all six splits are verified from the Drive side.**

---
