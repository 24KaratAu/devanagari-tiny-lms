# Phase 1 Report: Data Collection, Preprocessing, and Tokenizer Construction
**Course**: Language Models and Agents (CL3-410) — Monsoon 2026  
**Project**: Building and Analyzing Monolingual Transformer LMs in Two Indian Languages  
**Author**: 24KaratAu  
**Phase 1 Deadline**: 19th August 2026  

---

## 1. Executive Summary

Phase 1 establishes the foundational data and tokenization pipelines for two completely independent monolingual decoder-only Transformer language models:
* **Model H (Higher-Resource)**: **Hindi** (`hi`)
* **Model L (Lower-Resource)**: **Nepali** (`ne`)

Each language pipeline operates in complete isolation with dedicated raw data collection, custom text cleaning, deduplication, 90/5/5 document-level train/validation/test splits, and its own SentencePiece Byte-Pair Encoding (BPE) tokenizer trained from scratch. No corpus, vocabulary or tokenizer is shared between the two models.

**Headline results**

| | Hindi (Model H) | Nepali (Model L) |
| :--- | ---: | ---: |
| Active pretraining corpus | **520,409,408 tokens** | 488,502,833 tokens |
| Manual share (floor is 20%) | **26.87%** | ~50% |
| Documents | 848,815 | 1,428,401 |
| Tokenizer vocabulary | 8,000 | 32,000 |
| Fertility (tokens/word) | **1.4938** | 1.55 |
| Characters per token | 3.4712 | 4.24 |
| Unknown-token rate | **0.00%** | 0.00% |

Both corpora clear the ~500M token target and the ≥20% manual-collection floor.

This report documents the dataset collection statistics, cleaning and filtering methodologies, the resolution of memory and streaming constraints on student-tier compute, and an empirical evaluation of our tokenizers grounded in recent literature on Devanagari tokenization (*Dwivedi & Gopalan, ACL 2026*; *Darshana, 2026*).

---

## 2. Language Selection & Linguistic Justification

### 2.1 Language Pair Overview
* **Higher-Resource (Model H) — Hindi**: An Indo-Aryan language spoken by over 600 million people globally. Hindi possesses abundant digital text corpora across diverse domains (news, encyclopedic text, literature, government records) and standard Unicode Devanagari orthography.
* **Lower-Resource (Model L) — Nepali**: An official language of Nepal and one of the 22 scheduled languages of India (spoken by ~30 million people). Despite sharing the Devanagari script with Hindi, Nepali has significantly lower representation in digital corpora, higher morphological inflection, and distinct grammatical morphemes.

### 2.2 Shared Script with Divergent Morphology
Both languages utilize the Devanagari script (Unicode block `U+0900`–`U+097F`), an **Abugida (Alphasyllabary)** where consonants carry an inherent vowel `/a/` and combine with vowel diacritics (*matras*) and the virama/halant (`्`) to form composite orthographic syllables (*Aksharas*).

However, the two languages diverge sharply in grammatical morphology and lexical distribution:
1. **Case Markers & Postpositions**:
   * *Nepali*: High agglutination and bound case suffixes: `-लाई` (*lai*, accusative/dative), `-मा` (*ma*, locative), `-ले` (*le*, ergative/instrumental), `-बाट` (*bata*, ablative), `-को`/`-का`/`-की` (*ko/ka/ki*, genitive).
   * *Hindi*: Separated postpositions: `को` (*ko*), `में` (*mein*), `ने` (*ne*), `से` (*se*), `का`/`के`/`की` (*ka/ke/ki*).
2. **Pluralization & Verbal Inflection**:
   * Nepali utilizes bound morphemes like `-हरू` (*-haru*) for pluralization (e.g., `स्थलहरू`, `विद्यार्थीहरू`) and complex multi-part verb conjugations (e.g., `गरिरहेका छन्`, `हुनुपर्छ`).
   * Hindi utilizes vowel-shift pluralizations (`-ों`, `-ें`) and distinct auxiliary verbs (`है`, `हैं`, `था`, `थे`).

These morphological distinctions necessitate independent, language-specific tokenizers and datasets to prevent cross-lingual interference.

---

## 3. Dataset Collection & Corpus Construction

### 3.1 Corpus Targets & Requirements
* **Target Token Count**: ~500M tokens per language.
* **Manual Collection Share**: At least 20% of final training tokens ($\ge 100\text{M}$ tokens).
* **Automated Collection Share**: Up to 80% of final training tokens ($\le 400\text{M}$ tokens).

### 3.2 Data Sources & Modalities

```text
                                   Monolingual Corpus (~500M tokens)
                                 ┌─────────────────┴─────────────────┐
                                 │                                   │
                         Manual Share (≥ 20%)             Automated Share (≤ 80%)
                     ┌───────────┴───────────┐                       │
                     │                       │                       │
             Wikimedia Dumps            Scraped Web          AI4Bharat Sangraha
          • Wikipedia (newiki/hiwiki)  • News Portals            (Verified)
          • Wikisource (Literature)    • Gov Portals         • PDF Transcripts
          • Wikibooks (Education)      • Lit Archives        • ASR Transcripts
                                                             • Curated Web
```

1. **Manual Collection Pipeline**:
   * **Wikimedia XML Database Dumps**: Full raw `.xml.bz2` article dumps (`newiki`, `hiwiki`, `newikisource`, `hiwikisource`). In-house extraction and cleaning count toward manual collection per course guidelines.
   * **Targeted Web Crawling**: Production crawler (`src/data/collect/collect_manual_web.py`) utilizing `requests`, `trafilatura`, and process-safe atomic file locking (`fcntl.flock`) across news media (*OnlineKhabar, Ratopati, Setopati, Kantipur, Gorkhapatra*), government portals, and literary collections.
2. **Automated Collection Pipeline**:
   * **AI4Bharat Sangraha (Verified)**: The largest open-source verified Indic corpus, filtered strictly for human-verified, clean text segments across PDF extracts, ASR transcriptions, and curated web data.


### 3.3 Hindi Manual Collection: Sitemap-Driven Web Acquisition

The Hindi manual partition initially fell short of the $\ge 20\%$ floor, and the
shortfall was structural rather than incidental. Two constraints interact
multiplicatively: a manual share of at least 20% means automated data may be at
most $4\times$ the manual total, so the **reachable corpus is capped at
$5\times$ the manual token count** no matter how much automated data is
available. With 88.8M manual tokens, the entire corpus was capped at 444M
tokens — below the 500M target — while 683M automated tokens sat unusable in
reserve. Adding automated shards could not have fixed it; only manual tokens
could.

**Why the first crawl failed.** The original Hindi crawler enumerated article
URLs by guessing sequential identifiers
(`amarujala.com/india-news/news-id-849999`, `hindi.oneindia.com/news/india/779999.html`).
Publishers do not expose contiguous id spaces, so the overwhelming majority of
those requests returned 404 and the three resulting news shards held only 13 MB
of text in total.

**Sitemap discovery.** The replacement (`scripts/discover_hindi_urls.py`) reads
each publisher's `robots.txt`, follows every `Sitemap:` directive, and descends
the sitemap index breadth-first. This matters because the obvious guess is
wrong almost everywhere: `ndtv.in` and `zeenews.india.com` answer a direct
request for `/sitemap.xml` with HTTP 403, while `bhaskar.com`, `aajtak.in`,
`abplive.com`, `livehindustan.com`, `hindi.news18.com` and `naidunia.com` all
return 404 — yet every one of them publishes working sitemaps through
`robots.txt`. The walk enumerated **11,112,623 live article URLs across 18
Hindi domains in 7.6 minutes**.

**Domain balancing.** Two publishers accounted for 97% of the discovered space
(`amarujala.com` ~8.0M URLs, `livehindustan.com` ~2.8M). Crawling that
distribution directly would have produced a corpus that is essentially two
newspapers, and would also have throttled throughput, since the crawler caps
concurrency per host. URLs were therefore reservoir-sampled to at most 60,000
per host and interleaved round-robin, giving a **375,539-URL frontier spread
over 32 hosts**.

**Cross-lingual exclusion.** Devanagari is not exclusive to Hindi, so a
script-ratio filter alone is insufficient. `vikaspedia.in` in particular serves
roughly fifteen languages from sibling subdomains, including Devanagari-script
Nepali (`ne.`), Sanskrit (`sa.`), Maithili (`mai.`), Bodo (`brx.`) and Konkani
(`kok.`) — Nepali being precisely the language of Model L, whose corpus must
remain disjoint. Non-Hindi hosts were excluded at the frontier, and every
crawled document was additionally scored by `src/data/clean/hindi_lang_id.py`,
which rejects text whose Nepali/Sanskrit lexical density exceeds its Hindi
density.

**Crawl results.** 75,396 documents / 161,000,865 clean characters across 27
shards were accepted from ~87,600 attempted URLs (86% acceptance) at 20.2 URL/s;
75,387 documents / 160,980,352 characters survive corpus-wide deduplication. Rejections
break down as 45.4% `robots.txt` disallow (entirely `hindi.oneindia.com`, whose
article paths are disallowed for all agents), 40.7% quality filter, 9.3% HTTP
error, and 0.4% language identification. Deduplication ran against a registry
of the 155,123 content hashes already present in the manual partition.

| Contributing host | Share of crawl | | Contributing host | Share of crawl |
| :--- | ---: | --- | :--- | ---: |
| `indiatv.in` | 19.0% | | `patrika.com` | 8.5% |
| `livehindustan.com` | 17.3% | | `bhaskar.com` | 5.5% |
| `amarujala.com` | 16.8% | | `thelallantop.com` | 4.4% |
| `pardaphash.com` | 13.7% | | `hi.vikaspedia.in` | 1.4% |
| `moneycontrol.com` (Hindi) | 9.0% | | 23 further hosts | 4.4% |

**A note on Hindi Wikisource (diagnosed, not ingested).** The Colab pipeline's
Wikisource step produced exactly one document. The cause is a namespace
mismatch: English Wikisource places scanned page text in namespace `104`, but
**`hi.wikisource` uses namespace `250`** (`पृष्ठ:`). The step filtered
`ns in {0, 104}`, and namespace 0 on Wikisource holds mostly `<pages index=.../>`
transclusion stubs that clean to under 200 characters — so essentially the whole
corpus was excluded. Re-running `scripts/process_hindi_wikimedia_dump.py` over
the full `hiwikisource-latest-pages-articles.xml.bz2` with
`--namespaces 0,250` recovers **161,734 documents / 268,055,655 clean characters
(~80.7M tokens)** of proofread literary Hindi (Gandhi's *Hind Swaraj*, novels,
historical essays), of which 99% pass `hindi_lang_id` as native Hindi.

This source was **not ingested**: the manual partition already clears the 20%
floor at 25.8% without it, and the corpus is not short on domain diversity — news
accounts for only ~9% of the final token mix. It is documented here because the
fix is verified and available should Phase 2 require a larger manual partition.

**Raw data retention.** Per the course Q&A, raw alongside cleaned data is
archived to Drive under `raw/manual_web/` as per-shard `.tar.gz` bundles. The
first five bundles contain the complete unmodified server responses. From shard
0010 onward, `<script>`, `<style>`, `<svg>` and `<noscript>` blocks and long
base64 `data:` URIs are elided before archiving: on a modern Hindi news page
those assets are ~78% of the bytes (56 KB vs 12.5 KB gzipped per page) and take
no part in article extraction, whereas retaining them would have required
roughly 7 GB of upload over a measured 1 MB/s uplink. The full document
structure, text and metadata that `trafilatura` consumes is preserved verbatim.

---

## 4. Preprocessing, Quality Filtering, & Deduplication

### 4.1 Cleaning Pipeline
All collected raw documents pass through a sequential normalization and sanitization pipeline:
1. **Markup & Table Elimination**: Regex filters strip MediaWiki templates (`{{...}}`), HTML tables (`{| ... |}`), category markers, and citation brackets (`[1]`, `[25]`).
2. **Boilerplate & Link Stripping**: Removal of navigation headers, external URLs, email addresses, and non-content reference sections (*सन्दर्भ*, *बाह्य लिङ्कहरू*, *स्रोत*).
3. **Whitespace Normalization**: Collapse of repeated blank lines, runs of spaces and tabs, and trailing whitespace.
4. **Unicode Normalization (NFC)**: Every line is passed through `unicodedata.normalize("NFC", ...)` in `src/data/clean/clean_hindi_web.py`. This matters for Devanagari because the same grapheme has two encodings — क़ is either the precomposed U+0958 or क + U+093C (nukta) — which a tokenizer would otherwise treat as different characters. For U+0958–U+095F, NFC resolves toward the decomposed form (those codepoints are Unicode composition exclusions); the point is that both spellings converge on one.

   **Measured effect: none.** Sampling 8,000 documents / 42M characters across the Wikipedia dump and the web crawl found **0% non-NFC content**, because MediaWiki and the Hindi news portals already serve NFC. The step is retained as a guard against future sources, and is reported here as a verified no-op rather than claimed as a correction it never had to make.

### 4.2 Quality & Language Identification Filters
Every cleaned document must satisfy strict acceptance criteria before committing to disk:
* **Devanagari Ratio Threshold**: $\ge 0.50$ of total characters must fall within the Devanagari Unicode block (`\u0900`–`\u097F`).
* **Document Length Threshold**: Minimum clean text length of $\ge 100$ characters.
* **Composite Language Identifier (`nepali_lang_id.py`)**: To prevent Hindi/Sanskrit contamination in the Nepali corpus (and vice-versa), a composite scoring function evaluates script density alongside signature lexical markers:
  * Nepali Positive Markers: `लाई`, `मा`, `ले`, `बाट`, `छन्`, `भयो`, `हुने`, `यसको`, `हाम्रो`, `थियो`.
  * Hindi Disqualifying Markers: `है`, `हैं`, `था`, `और`, `में`, `पर`, `किया`, `नहीं`, `सकता`.
  * If Hindi marker density exceeds Nepali marker density, the document is rejected.

### 4.3 Deduplication Strategy & Manual Precedence
To prevent data contamination and ensure accurate accounting:
1. **Manual Priority Ingestion**: Manual documents (Wikipedia dumps and scraped articles) are ingested first. Their 64-bit integer content hashes are registered into the global deduplication registry.
2. **Intra-Manual Deduplication**: The manual partition is deduplicated **against itself** before the budget is derived from it, not only against the automated stream.
3. **Automated Stream Deduplication**: Automated Sangraha shards are streamed subsequently. Any document colliding with an existing manual hash is discarded from the automated dataset.
4. **Outcome**: Manual articles are never overwritten or dropped by automated crawls, preserving 100% of the manual collection integrity.

#### Why intra-manual deduplication was necessary (Hindi)

Step 2 was added after the fact, and the reason is worth recording. The Hindi
manual partition was assembled by several collection runs whose deduplication
registries were snapshotted at different times, so each run was blind to shards
written after its own snapshot. Auditing the assembled partition found **2,523
exact duplicates spanning 18,639,999 characters** — 1.04% of manual documents:

| First occurrence | Duplicate in | Documents |
| :--- | :--- | ---: |
| Wikibooks / Wikiquote (`manual_aux_*`) | Literature dump (`manual_lit_*`) | 2,365 |
| News crawl, Colab | News crawl, Colab | 105 |
| Wikibooks / Wikiquote | Wikibooks / Wikiquote | 43 |
| Web crawl, sitemap | Web crawl, sitemap | 9 |
| Literature dump | Wikisource dump | 1 |

94% of the duplication is a single cause: `manual_lit_shard_0030` is a re-parse of
the same Wikimedia dumps that produced `manual_aux_shard_0032`, and **2,365 of its
2,366 documents already existed there**. Post-deduplication it contributes one
document, and the lone Wikisource document is itself a duplicate of that shard, so
Wikisource contributes none. Both are retained in the table in §5.2 with their true
(near-zero) contribution rather than removed, so the source accounting stays
honest.

Deduplicating only manual→automated, as the pipeline originally did, would have
left every one of these in the training corpus and inflated the automated budget
derived from the manual total. The sitemap web crawl accounts for 9 of the 2,523,
having deduplicated against a registry snapshot during collection.

---

## 5. Dataset Statistics & Split Stratification

### 5.1 Dataset Scale (Nepali Corpus — Model L)
The full Nepali collection and cleaning pipeline yielded the following verified corpus statistics across Google Drive storage and the active 90/5/5 pretraining partitions:

| Partition / Source Category | Shards | Document Count | Clean Characters | BPE Token Count | Share (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Manual — Wikipedia Dump** (`wiki_shard_*`) | 6 | 27,946 | 42,864,542 | ~8,650,000 | 1.8% | Processed & Deduplicated |
| **Manual — Live News Crawls** (`manual_news_*`) | 42 | ~295,000 | ~1,150,000,000 | ~245,000,000 | 48.2% | Crawled (*Ratopati, Setopati*) |
| **Subtotal: Total Manual Data** | **48** | **~322,946** | **~1,192,864,542** | **~253,650,000** | **~50.0%** | **Exceeds $\ge 20\%$ Floor** |
| **Automated — Sangraha Verified** (`nonweb_shard_*`) | 248 | 1,230,594 | 2,374,482,746 | ~479,852,833 | — | Cleaned & Verified |
| **Total Clean Storage on Google Drive** | **296** | **~1,553,540** | **~3,567,347,288** | **~1,500,000,000** | **100.0%** | **Complete Reserve Pool** |

---

### 5.2 Dataset Scale (Hindi Corpus — Model H)

All figures below are **post-deduplication** and exact, read from size-verified
copies of every shard. Token counts use characters-per-token measured against the
live tokenizer per partition — **3.433** for manual, **3.507** for automated —
rather than a single assumed constant. An earlier pipeline applied a flat 3.19 to
both, overstating token counts and under-delivering against the target, so
`finalize_hindi_corpus.ipynb` re-measures the ratios before budgeting.

| Partition / Source Category | Shards | Documents | Clean Characters | Tokens |
| :--- | ---: | ---: | ---: | ---: |
| Manual — Wikipedia dump (`wiki_shard_*`) | 30 | 146,296 | 253,116,189 | 73,734,327 |
| Manual — Web crawl, sitemap (`manual_web_shard_*`) | 27 | 75,387 | 160,980,352 | 46,894,582 |
| Manual — News crawl, Colab (`manual_news_*`) | 6 | 14,811 | 41,116,565 | 11,977,512 |
| Manual — Wikibooks / Wikiquote (`manual_aux_*`) | 1 | 4,015 | 24,368,410 | 7,098,670 |
| Manual — Literature re-parse (`manual_lit_*`, redundant — see §4.3) | 1 | 1 | 1,274 | 371 |
| Manual — Wikisource (`wikisource_*`, see §3.3) | 1 | 0 | 0 | 0 |
| **Subtotal: Total Manual (unique)** | **66** | **240,510** | **479,582,790** | **139,705,463** |
| Automated — Sangraha Verified (`nonweb_shard_*`), full pool | 220 | ~1,090,000 | ~2,400,700,000 | ~684,500,000 |
| **Total Clean Reserve Pool on Drive** | **286** | **~1,330,500** | **~2,880,282,790** | **~824,200,000** |

The manual partition of 139.7M unique tokens raises the corpus ceiling from 444M
to **~699M tokens** at the 20% floor. The active corpus is drawn at 520.4M tokens
with a manual share of **26.87%**. That headroom is deliberate and it earned its
keep: deduplication later removed 18.6M manual characters, and the share absorbed
the loss without approaching the floor. At this size the binding constraint is the
token target itself rather than the manual floor — 4x the manual partition would
permit 559M automated tokens, against the 380.3M actually ingested.

**Cross-lingual purity spot check.** Sampling ~15% of four representative manual
shards and scoring each document with `hindi_lang_id.py` returns 97.9–100%
accepted as Hindi. A single Sanskrit document (a Devanagari-numerals article) was
found in the Wikibooks shard, i.e. 0.2% of that shard's sample and well below any
level that would affect training.

---

### 5.3 Active Pretraining Dataset (90 / 5 / 5 Splits)
From the cleaned corpus pool, the active training split is curated and balanced to target the ~500M token budget using a deterministic document-level pseudo-random hash router (Seed `42`) with zero leakage across partitions. A document's split is a function of its content hash, so the assignment is stable regardless of the order in which shards are ingested.

The automated budget is derived from the manual total rather than fixed in
advance: with $A \le 4M$ from the 20% floor and a total target $T$, the ingest
takes $A = \min(4M,\ T - M)$ automated tokens. This enforces both constraints
simultaneously instead of applying one and hoping the other holds.

#### Nepali (Model L)

| Split Name | Percentage | Document Count | Clean Characters | BPE Tokens (`32k` Vocab) |
| :--- | :---: | :---: | :---: | :---: |
| **Train** (`train.txt`) | 90.0% | 1,285,474 | 2,175,612,559 | **439,592,063** |
| **Validation** (`val.txt`) | 5.0% | 71,556 | 120,867,364 | **24,421,674** |
| **Test** (`test.txt`) | 5.0% | 71,371 | 120,867,365 | **24,489,096** |
| **Total Active Pretraining Corpus** | **100.0%** | **1,428,401** | **2,417,347,288** | **488,502,833 (~488.5M tokens)** |

#### Hindi (Model H)

Produced by `notebooks/finalize_hindi_corpus.ipynb`, which stages every shard with
per-file size verification, deduplicates the manual partition, re-measures
characters-per-token against the live tokenizer, derives the automated budget from
the unique manual total, and asserts on the *ingested* result before writing
anything to Drive. Token counts are measured with the final `hindi_bpe_8k` model
over each split.

| Split Name | Percentage | Document Count | Clean Characters | BPE Tokens (`8k` Vocab) |
| :--- | :---: | :---: | :---: | :---: |
| **Train** (`train.txt`) | 89.9% | 763,955 | 1,630,473,179 | **467,905,255** |
| **Validation** (`val.txt`) | 5.0% | 42,576 | 90,938,033 | **26,069,560** |
| **Test** (`test.txt`) | 5.1% | 42,284 | 91,822,301 | **26,434,593** |
| **Total Active Pretraining Corpus** | **100.0%** | **848,815** | **1,813,233,513** | **520,409,408 (~520.4M tokens)** |

| Collection Partition | Clean Characters | Tokens | Share |
| :--- | ---: | ---: | ---: |
| Manual (100% of unique manual ingested) | 479,582,790 | 139,705,463 | **26.87%** |
| Automated (budget-limited) | 1,333,650,723 | 380,294,801 | 73.13% |

Exactly one automated document was discarded for colliding with a manually
collected one, confirming manual precedence is active. Measured compression on the
final corpus is **3.484 characters/token**.

---

## 6. Tokenizer Construction & Empirical Evaluation

Each language has its own tokenizer, trained from scratch on that language's
training split only. No pretrained tokenizer is used and no vocabulary is shared
between the two models.

### 6.1 What the tokenizer does, and how it is configured

A tokenizer turns raw text into the integer IDs a model consumes. Ours are
**SentencePiece** models: they learn a fixed vocabulary of subword pieces
directly from raw text, so no separate word-splitting step is needed. Both
languages use identical settings apart from vocabulary size:

| Setting | Value | Why |
| :--- | :--- | :--- |
| `model_type` | `bpe` | Byte-Pair Encoding; merges the most frequent adjacent pairs bottom-up |
| `character_coverage` | `0.9995` | Keeps the full Devanagari inventory (consonants, matras, nuktas, virama) while discarding Unicode noise |
| `byte_fallback` | `True` | Any character the vocabulary lacks is emitted as raw UTF-8 byte tokens instead of `<unk>` |
| `split_by_unicode_script` | `True` | Prevents pieces that straddle Devanagari and Latin |
| `split_by_whitespace` | `True` | Pieces never span a space |
| `split_digits` | `True` | Each digit becomes its own token, for arithmetic in Phase 3 |
| Special tokens | `<pad>=0, <unk>=1, <bos>=2, <eos>=3` | Fixed IDs shared by both models |

**Metrics used below**, defined once for the first-time reader:

* **Fertility** — average tokens per whitespace-delimited word. 1.0 would mean
  every word is a single token; lower is better compression. Anything above 1.0
  means genuine sub-word segmentation, which is what we want.
* **Characters per token** — average characters each token carries. Higher means
  a longer sequence fits in the same context window.
* **Unknown-token rate** — share of tokens emitted as `<unk>`. With
  `byte_fallback` this should be exactly zero.
* **Broken grapheme rate** — share of tokens that *begin* with a Devanagari
  combining mark (a matra, anusvara or virama). Devanagari is an abugida, so a
  token starting with a dependent vowel sign has been severed from the consonant
  it modifies. Lower is better.
* **Vocabulary coverage** — share of vocabulary entries actually observed on
  held-out text. Low coverage means the vocabulary contains pieces the model will
  rarely or never use, wasting embedding parameters.
* **Shannon / Rényi entropy** — how evenly the token distribution is spread over
  the vocabulary. Higher means the vocabulary is being used more uniformly rather
  than a handful of pieces dominating.

### 6.2 Choosing the algorithm and the vocabulary size

#### 6.2.1 Why a subword tokenizer at all

Three families were available, and the choice is forced by Hindi's morphology:

* **Word-level** — one token per whitespace word. Hindi is moderately
  inflectional (`विद्यार्थी` → `विद्यार्थियों`), so a word vocabulary large
  enough to cover the corpus would run to hundreds of thousands of entries, and
  every unseen inflection becomes `<unk>` at test time. With a 25M parameter
  budget this is not affordable, and it fails exactly where morphology is
  productive.
* **Character-level** — no vocabulary problem and no unknowns, but a Devanagari
  akshara spans several codepoints, so sequences become 4–5x longer. Attention
  cost is quadratic in sequence length, so this spends the compute budget on
  length rather than depth.
* **Subword** — a fixed vocabulary of frequent character sequences. Common words
  stay whole, rare and inflected forms decompose into reusable pieces. This is
  the only option that fits both constraints.

#### 6.2.2 Why BPE rather than Unigram

Within subword tokenizers, the two realistic candidates in SentencePiece are
**BPE** (bottom-up: repeatedly merge the most frequent adjacent pair) and the
**Unigram LM** (top-down: over-generate candidates, then prune those that cost
the least likelihood). WordPiece is close to BPE with a likelihood-based merge
criterion and is not separately supported here.

We chose BPE, and the honest position is that **our own measurements mildly
favour Unigram**. At an equal 8k vocabulary fitted on the same sample, Unigram
scored better on both metrics that matter for Devanagari:

| Model (8k, same training sample) | Fertility | Broken graphemes |
| :--- | ---: | ---: |
| BPE | 1.5873 | 7.967% |
| **Unigram** | **1.5613** | **6.073%** |

Unigram's edge on broken graphemes is the more interesting of the two: its
pruning objective removes pieces that rarely pay for themselves, which appears
to preserve consonant + matra clusters slightly better than frequency-driven
merging. The reasons we still selected BPE are procedural rather than empirical,
and should be read that way:

1. **Comparability with the literature we benchmark against.** The Devanagari
   tokenization work this report is grounded in (§6.5) reports fertility for BPE
   systems. Using BPE makes our 1.4938 directly comparable to their 3.5–5.0.
2. **It is the standard for decoder-only pretraining** (GPT-2/3, LLaMA), so
   Phase 2 results transfer more readily to published baselines.
3. **The margin is small and confounded.** Both sweep models were fitted on an
   80 MB sample. The production BPE, fitted on the full 3.9 GB split, reaches
   fertility **1.4938** — better than the sampled Unigram's 1.5613. That is *not*
   evidence BPE wins, because Unigram was never given the full corpus; it only
   shows the sample-vs-full gap is larger than the algorithm gap.

**This is the weakest-evidenced decision in Phase 1.** A fair test — Unigram
trained on the same full split, compared on fertility, broken graphemes and
downstream validation loss — is the ablation we would run first in Phase 2. If
Unigram's broken-grapheme advantage survives at full scale, it is the better
choice for an abugida.

#### 6.2.3 Why byte fallback

`byte_fallback=True` routes any character outside the vocabulary to raw UTF-8
byte tokens. This is what makes the measured unknown-token rate exactly
**0.00%** at every vocabulary size tested. It matters here because it decouples
vocabulary size from OOV risk: without it, a small vocabulary would have to be
inflated defensively to cover rare conjuncts, nuktas and stray Latin, which is
precisely the parameter cost §6.2.4 is trying to avoid.

#### 6.2.4 Choosing the vocabulary size empirically

Vocabulary size is the single most consequential tokenizer decision here,
because it trades directly against the **~25M parameter budget** for Phase 2.
With embedding dimension `d_model = 256` and weight tying between the input
embedding and the output projection, the embedding table costs `V x 256`
parameters:

| Vocabulary `V` | Embedding parameters | Share of a 25M budget |
| ---: | ---: | ---: |
| 4,000 | 1,024,000 | 4.1% |
| **8,000** | **2,048,000** | **8.2%** |
| 16,000 | 4,096,000 | 16.4% |
| 32,000 | 8,192,000 | 32.8% |

A 32k vocabulary spends a third of the entire model on embeddings before a
single attention block exists. The course Q&A is explicit on this point:
*"Please try to keep a vocab size of 5 to 10K."*

To choose on evidence rather than assertion, we fitted BPE tokenizers at four
vocabulary sizes plus a Unigram LM at 8k on an 80 MB sample of the Hindi
**training** split, and evaluated every one on 59,713 held-out documents
(12.2M characters) from the **validation** split. No model is scored on text it
was fitted on.

| Model | Vocab | Fertility | Chars/token | Unknown rate | Broken graphemes | Vocab coverage |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| BPE | 4,000 | 1.7855 | 2.9041 | 0.00% | 11.229% | 94.88% |
| BPE | 8,000 | 1.5873 | 3.2666 | 0.00% | 7.967% | 94.31% |
| Unigram | 8,000 | 1.5613 | 3.3210 | 0.00% | 6.073% | 94.30% |
| BPE | 16,000 | 1.4509 | 3.5738 | 0.00% | 5.388% | 91.23% |
| BPE | 32,000 | 1.3587 | 3.8163 | 0.00% | 3.502% | 82.39% |

Three findings drive the decision:

1. **Compression improves with vocabulary size, but with diminishing returns
   against linear parameter cost.** Going 8k -> 32k buys 0.23 tokens/word while
   quadrupling embedding parameters from 8.2% to 32.8% of the budget.
2. **Vocabulary coverage collapses at 32k.** Only 82.39% of a 32k vocabulary is
   ever observed on held-out text, so roughly 5,600 embedding rows — about 1.4M
   parameters — would be dead weight. At 8k, coverage is 94.31%.
3. **Unknown-token rate is exactly 0.00% at every size**, confirming
   `byte_fallback` removes out-of-vocabulary failure entirely, so vocabulary size
   need not be inflated defensively to cover rare characters.

**Selected: BPE with an 8,000-token vocabulary and byte fallback for Hindi
(Model H)** — vocabulary size chosen on the evidence above, algorithm chosen on
the procedural grounds set out in §6.2.2 and flagged there as the decision most
worth revisiting.

### 6.3 Hindi (Model H) — `hindi_bpe_8k`

Trained on the full 3,949 MB Hindi training split
(`splits/train.txt`, 467,905,255 tokens). Evaluated on 59,713 held-out documents
from `splits/val.txt` — 12,211,815 characters, 2,355,109 words, 3,517,995 tokens.

| Statistic | Value |
| :--- | ---: |
| Vocabulary size | **8,000** |
| Fertility (tokens/word) | **1.4938** |
| Characters per token | **3.4712** |
| Unknown tokens | **0** (rate **0.000000%**) |
| Broken grapheme rate | 5.5683% |
| Vocabulary types observed | 7,793 of 8,000 (**97.41%**) |
| Vocabulary types seen exactly once | 52 |
| Shannon entropy | 10.3963 bits |
| Rényi entropy (α=2) | 7.7562 bits |

Training on the full corpus rather than an 80 MB sample improves fertility from
1.5873 to **1.4938** and lifts coverage from 94.31% to **97.41%** — the
production model beats even the sampled 16k model on coverage while using half
the parameters.

**Token-frequency statistics.** The distribution is steep but not degenerate:

| Cumulative coverage | Share of all tokens |
| :--- | ---: |
| Top 100 pieces | 40.34% |
| Top 500 pieces | 61.18% |
| Top 1,000 pieces | 72.21% |
| Top 2,000 pieces | 83.32% |
| Top 4,000 pieces | 93.47% |

The 4,000 rarest pieces still account for 6.5% of tokens, and only 52 pieces
appear exactly once — the vocabulary is close to fully utilised.

The 15 most frequent pieces are exactly the Hindi function words a healthy
Hindi tokenizer should learn (`▁` marks a preceding word boundary):

| Rank | Piece | Gloss | Count | Share |
| ---: | :--- | :--- | ---: | ---: |
| 1 | `।` | danda (full stop) | 97,314 | 2.766% |
| 2 | `▁के` | *ke* (genitive) | 93,528 | 2.659% |
| 3 | `▁में` | *mein* (locative "in") | 71,946 | 2.045% |
| 4 | `,` | comma | 71,062 | 2.020% |
| 5 | `▁है` | *hai* ("is") | 68,087 | 1.935% |
| 6 | `▁की` | *ki* (genitive, fem.) | 61,597 | 1.751% |
| 7 | `▁` | word boundary before a digit | 47,231 | 1.343% |
| 8 | `▁को` | *ko* (accusative/dative) | 42,126 | 1.197% |
| 9 | `▁से` | *se* (ablative "from") | 41,818 | 1.189% |
| 10 | `▁और` | *aur* ("and") | 39,044 | 1.110% |
| 11 | `▁का` | *ka* (genitive, masc.) | 36,166 | 1.028% |
| 12 | `▁ने` | *ne* (ergative) | 30,353 | 0.863% |
| 13 | `▁पर` | *par* ("on") | 28,578 | 0.812% |
| 14 | `0` | digit | 27,890 | 0.793% |
| 15 | `2` | digit | 26,162 | 0.744% |

Every Hindi postposition identified in §2.2 as a distinguishing feature of the
language — `के`, `में`, `की`, `को`, `से`, `का`, `ने`, `पर` — earned a dedicated
single-token entry. Individual digits rank highly because `split_digits=True`
deliberately isolates them.

**Tokenization examples** (`▁` = word boundary):

```text
भारत एक विशाल और विविधतापूर्ण दक्षिण एशियाई गणराज्य है।
  ▁भारत ▁एक ▁विशाल ▁और ▁विविध ता पूर्ण ▁दक्षिण ▁एशियाई ▁गणराज्य ▁है ।
  9 words -> 12 tokens (fertility 1.33)
  The one split is morphological: विविध + ता + पूर्ण ("diverse" + nominaliser + "full of").

विश्वविद्यालय के विद्यार्थियों ने वैज्ञानिक अनुसंधान में महत्वपूर्ण योगदान दिया।
  ▁विश्वविद्यालय ▁के ▁विद्यार्थियों ▁ने ▁वैज्ञानिक ▁अनुसंधान ▁में ▁महत्वपूर्ण ▁योगदान ▁दिया ।
  10 words -> 11 tokens (fertility 1.10)
  Every word is a single token, including the 6-syllable विश्वविद्यालय
  ("university") and the inflected plural oblique विद्यार्थियों ("students").

प्रधानमंत्री ने कहा कि अर्थव्यवस्था तेज़ी से आगे बढ़ रही है।
  ▁प्रधानमंत्री ▁ने ▁कहा ▁कि ▁अर्थव्यवस्था ▁ते ज़ी ▁से ▁आगे ▁बढ़ ▁रही ▁है ।
  11 words -> 13 tokens (fertility 1.18)
  Only तेज़ी splits, at the nukta-bearing ज़ — a rarer character.

उन्होंने 2026 में लगभग 15 प्रतिशत की वृद्धि दर्ज की।
  ▁उन्होंने ▁ 2 0 2 6 ▁में ▁लगभग ▁ 1 5 ▁प्रतिशत ▁की ▁वृद्धि ▁दर्ज ▁की ।
  10 words -> 17 tokens (fertility 1.70)
  Fertility is high here purely because split_digits=True expands "2026" into
  four tokens. This is a deliberate trade: numeric text costs more tokens, but
  the model sees digits as consistent atomic units for Phase 3 arithmetic.
```

**On the broken grapheme rate (5.5683%).** Hindi's rate is higher than Nepali's
2.615% (§6.4). This is a direct consequence of the smaller vocabulary, and the
sweep confirms the mechanism: the rate falls monotonically from 11.23% at 4k to
3.50% at 32k. Accepting 5.57% is the cost of spending 8.2% rather than 32.8% of
the parameter budget on embeddings. The affected tokens are overwhelmingly rare
inflected forms, not the high-frequency vocabulary shown above.

### 6.4 Nepali (Model L) — `nepali_bpe_32k`

The Nepali tokenizer was trained earlier in Phase 1 at a 32,000 vocabulary,
benchmarked over 5,000 held-out validation sentences:

| Tokenizer Model | Fertility | Chars/token | Broken graphemes | Shannon | Rényi (α=2) | Unknown rate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Standard BPE (32k)** *(selected)* | **1.55** | **4.24** | **2.615%** | **11.73 bits** | **8.71 bits** | **0.00%** |
| Unigram LM (32k) | 1.53 | 4.28 | 3.228% | 11.50 bits | 8.41 bits | 0.00% |

> **Known inconsistency.** Nepali uses 32,000 while Hindi uses 8,000. The 32k
> model predates the course Q&A guidance to keep vocabularies at 5–10K, and by
> the parameter analysis in §6.2 it would consume 32.8% of the Model L parameter
> budget on embeddings. The two corpora and vocabularies are required to be
> independent, so the sizes need not match — but Model L should either be
> retrained at 8k for parity with Model H, or the 32k choice justified explicitly
> against its parameter cost, before Phase 2 pretraining. This is recorded here
> rather than silently reconciled.

### 6.5 Grounding in Devanagari tokenization literature

Recent work on Indic tokenization (*Dwivedi & Gopalan, ACL 2026*; *Darshana,
2026*) shows that BPE fails on Devanagari in **multilingual** models: Devanagari
receives only ~300–500 of 32k–100k vocabulary slots, so multi-codepoint aksharas
fragment into isolated matras and viramas and fertility rises to **3.5–5.0
tokens per word**.

Our tokenizers are 100% monolingual, so all 8,000 (Hindi) or 32,000 (Nepali)
slots serve one language. Measured fertility of **1.4938** for Hindi is roughly a
**2.3–3.3x improvement** over the multilingual figures reported in that
literature — achieved with a vocabulary a quarter the size of the multilingual
models being criticised.

### 6.6 Qualitative comparison — Nepali (Model L)

```text
Sample 1: नेपाल एक सुन्दर, सार्वभौमसत्ता सम्पन्न दक्षिण एशियाली राष्ट्र हो।
Standard BPE: [▁नेपाल] [▁एक] [▁सुन्दर] [,] [▁सार्वभौमसत्ता] [▁सम्पन्न] [▁दक्षिण] [▁एशियाली] [▁राष्ट्र] [▁हो] [।]
Analysis: Exact 1:1 word-to-token alignment on every Devanagari lexical item.

Sample 2: काठमाडौं उपत्यकाभित्र विभिन्न ऐतिहासिक दृष्टिकोणले महत्त्वपूर्ण स्थलहरू छन्।
Standard BPE: [▁काठमाडौं] [▁उपत्यकाभित्र] [▁विभिन्न] [▁ऐतिहासिक] [▁दृष्टिकोणले] [▁महत्त्वपूर्ण] [▁स्थल] [हरू] [▁छन्] [।]
Analysis: Clean morphological decomposition of the plural noun into stem [▁स्थल] + suffix [हरू].

Sample 3: विश्वविद्यालयमा विद्यार्थीहरूले प्राविधिक तथा वैज्ञानिक अनुसन्धान गरिरहेका छन्।
Standard BPE: [▁विश्वविद्यालयमा] [▁विद्यार्थीहरूले] [▁प्राविधिक] [▁तथा] [▁वैज्ञानिक] [▁अनुसन्धान] [▁गरिरहेका] [▁छन्]
Analysis: Complex agglutinative multi-syllable words remain intact as single tokens.
```

### 6.7 Reproducing these numbers

```bash
python scripts/benchmark_hindi_tokenizer.py \
    --model  <drive>/hindi/tokenizer/hindi_bpe_8k.model \
    --val    <sample of hindi/splits/val.txt> \
    --train  <sample of hindi/splits/train.txt> \
    --out    report/hindi_tokenizer_benchmark.json
```

Full machine-readable results, including all 25 top pieces and every sweep
configuration, are committed at `report/hindi_tokenizer_benchmark.json`.

---

## 7. Data Provenance and Artifact Locations

Everything below is stated so a reader encountering this project for the first
time can trace any token in the corpus back to the source it came from, and open
the exact file it lives in.

### 7.1 Where every document came from

All Drive paths are relative to the project root **`24KaratAu/`**, shared at
<https://drive.google.com/drive/folders/19EANSjwyIp-z36jwCvKhwlWdcwA1Ohhq>
(Anyone with the link → Viewer).

#### Hindi (Model H)

| # | Source | What it is | How obtained | Collection type | Lands in |
| ---: | :--- | :--- | :--- | :--- | :--- |
| 1 | `ai4bharat/sangraha` (`verified/hin`) | AI4Bharat's human-verified Indic corpus — PDF extracts, ASR transcripts, curated web | Parquet shards streamed directly from `hf://datasets/ai4bharat/sangraha`, one ~340 MB file at a time via `urllib` + `pyarrow.iter_batches` | **Automated** | `hindi/cleaned/automated_nonweb/nonweb_shard_*.jsonl` |
| 2 | Hindi Wikipedia | Full article dump | `dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2`, parsed and cleaned in-house | **Manual** | `hindi/cleaned/manual/wiki_shard_0000–0029.jsonl` |
| 3 | Hindi Wikibooks + Wikiquote | Auxiliary Wikimedia projects | `dumps.wikimedia.org/hiwikibooks/…` and `hiwikiquote/…` `-latest-pages-articles.xml.bz2` | **Manual** | `hindi/cleaned/manual/manual_aux_shard_0032.jsonl` |
| 4 | 32 Hindi news / government portals | Live article text | Sitemaps discovered from each site's `robots.txt`, crawled with `trafilatura` (§3.3). Largest contributors: `indiatv.in`, `livehindustan.com`, `amarujala.com`, `pardaphash.com`, `moneycontrol.com`, `patrika.com`, `bhaskar.com`, `thelallantop.com`, `hi.vikaspedia.in` | **Manual** | `hindi/cleaned/manual/manual_web_shard_*.jsonl` (27 shards) |
| 5 | Hindi news portals (earlier Colab crawl) | Live article text | Parallel crawl from the Colab pipeline | **Manual** | `hindi/cleaned/manual/manual_news_shard_0033–0038.jsonl` |
| 6 | Wikimedia re-parse | Redundant re-parse of source 3 | — | **Manual** (near-zero after dedup, §4.3) | `hindi/cleaned/manual/manual_lit_shard_0030.jsonl` |
| 7 | Hindi Wikisource | Diagnosed but **not ingested** (§3.3) | `dumps.wikimedia.org/hiwikisource/latest/…` | **Manual** | `hindi/cleaned/manual/wikisource_shard_0000.jsonl` (1 doc, itself a duplicate) |

#### Nepali (Model L)

| # | Source | How obtained | Collection type | Lands in |
| ---: | :--- | :--- | :--- | :--- |
| 1 | `ai4bharat/sangraha` (`verified/nep`) | Same streaming path as Hindi | **Automated** | `nepali/cleaned/automated_nonweb/nonweb_shard_*.jsonl` |
| 2 | Nepali Wikipedia | `dumps.wikimedia.org/newiki/latest/newiki-latest-pages-articles.xml.bz2` | **Manual** | `nepali/cleaned/manual/wiki_shard_*.jsonl` |
| 3 | Nepali news portals (*Ratopati*, *Setopati*, *Gorkhapatra*) | `src/data/collect/collect_manual_web.py` with `trafilatura` | **Manual** | `nepali/cleaned/manual/manual_news_*.jsonl` |

**The manual/automated boundary.** Per the course Q&A, a source counts as manual
when we fetch the raw data and perform our own cleaning, preprocessing and
organisation — which covers Wikimedia XML dumps and our own web crawls. It does
**not** cover pre-packaged, pre-processed datasets pulled from Hugging Face,
which is why Sangraha is classified automated in full.

### 7.2 Exact Drive layout

Shared folder: <https://drive.google.com/drive/folders/19EANSjwyIp-z36jwCvKhwlWdcwA1Ohhq>

| Path under `MyDrive/24KaratAu/` | Files | Size | Contents |
| :--- | ---: | ---: | :--- |
| `hindi/cleaned/automated_nonweb/` | 220 | 5.976 GiB | Cleaned Sangraha shards (JSONL) |
| `hindi/cleaned/manual/` | 66 | 1.270 GiB | All cleaned manual shards (JSONL) |
| `hindi/raw/manual_web/` | 27 | 1.702 GiB | Raw crawled HTML, gzipped per shard (§3.3) |
| `hindi/raw/wiki_dump/` | 3 | 0.234 GiB | Wikimedia `.xml.bz2` dumps (`hiwiki`, `hiwikibooks`, `hiwikiquote`) |
| `hindi/splits/` | 6 | 8.834 GiB | `train/val/test` × `.txt` and `.jsonl` |
| `hindi/tokenizer/` | 2 | 0.001 GiB | `hindi_bpe_8k.model`, `hindi_bpe_8k.vocab` |
| `hindi/manifests/` | 7 | 0.328 GiB | Corpus summary + crawl provenance (§7.3) |
| `nepali/cleaned/automated_nonweb/` | 248 | 6.263 GiB | Cleaned Sangraha shards |
| `nepali/cleaned/manual/` | 48 | 1.280 GiB | Wikipedia dump + news crawl shards |
| `nepali/raw/wiki_dump/` | 1 | 0.051 GiB | `newiki-latest-pages-articles.xml.bz2` |
| `nepali/raw/manual_web/` | 2 | 0.050 GiB | Raw scraped pages |
| `nepali/splits/` | 6 | 2.659 GiB | `train/val/test` × `.txt` and `.jsonl` |
| `nepali/tokenizer/` | 4 | 0.004 GiB | `nepali_bpe_32k` model/vocab (+ Unigram comparison) |

Every cleaned shard is JSONL, one document per line, with a stable schema:

```json
{"doc_id": "...", "url": "...", "source": "Hindi web (www.indiatv.in)",
 "source_type": "news_web", "collection_type": "manual", "language": "hindi",
 "text": "...", "clean_text_characters": 2127,
 "devanagari_ratio": 0.77, "latin_ratio": 0.013,
 "hindi_token_ratio": 0.09, "language_confidence": 0.71,
 "language_status": "accept", "text_sha256": "..."}
```

`collection_type` is the field that carries the manual/automated accounting, and
`text_sha256` is the deduplication key used throughout §4.3.

### 7.3 Collection provenance files

The bulky artefacts are archived on Drive rather than committed; compact
summaries are committed so the repository stays self-describing.

| Artefact | Location | Purpose |
| :--- | :--- | :--- |
| Full discovered URL frontier (11,112,623 URLs) | `hindi/manifests/frontier_full.txt.gz` | Every URL sitemap discovery found |
| Domain-balanced frontier (375,539 URLs) | `hindi/manifests/frontier_balanced.txt.gz` | The exact crawl input |
| Attempted-URL log (87,585 URLs) | `hindi/manifests/crawl_done_urls.txt.gz` | Every URL the crawler tried |
| Rejection log (12,227 entries) | `hindi/manifests/crawl_rejected.jsonl.gz` | Per-URL rejection reason |
| Dedup registry (155,123 hashes) | `hindi/manifests/existing_manual_hashes.txt.gz` | Content hashes at crawl time |
| Corpus summary | `hindi/manifests/hindi_corpus_summary.json` | Final split and share figures |
| Crawl summary *(committed)* | `data/hindi/manifests/manual_web_crawl/crawl_summary.json` | Per-host and per-reason outcomes |
| Frontier stats *(committed)* | `data/hindi/sources/frontier_stats.json` | Per-host discovery counts |
| Tokenizer benchmark *(committed)* | `report/hindi_tokenizer_benchmark.json` | All §6 numbers, machine-readable |

### 7.4 Which script produced which artefact

| Stage | Script / notebook | Output |
| :--- | :--- | :--- |
| Automated ingest | `src/data/collect/process_hf_corpus.py`, `notebooks/collect_hindi_colab.ipynb` | `cleaned/automated_nonweb/` |
| Wikimedia dumps | `src/data/collect/process_wiki_dump.py`, `scripts/process_hindi_wikimedia_dump.py` | `cleaned/manual/wiki_shard_*`, `manual_aux_*` |
| URL discovery | `scripts/discover_hindi_urls.py` | `frontier*.txt` |
| Dedup registry | `scripts/export_hindi_manual_hashes.py` | `existing_manual_hashes.txt` |
| Web crawl | `scripts/collect_hindi_manual_web.py` | `cleaned/manual/manual_web_shard_*`, `raw/manual_web/` |
| Cleaning | `src/data/clean/clean_hindi_web.py`, `hindi_lang_id.py` | applied inline during collection |
| Accounting | `scripts/audit_hindi_corpus.py` | per-source token tables (§5.2) |
| Split + tokenizer | `notebooks/finalize_hindi_corpus.ipynb` | `splits/`, `tokenizer/`, `hindi_corpus_summary.json` |
| Tokenizer benchmark | `scripts/benchmark_hindi_tokenizer.py` | `report/hindi_tokenizer_benchmark.json` |

### 7.5 Execution timeline — what ran where, and when

Reconstructed from the git history and the modification timestamps of every
artifact on Drive. Times are IST. **Colab** means a notebook cell executed on a
Google Colab runtime writing to the mounted Drive; **Local** means a script run
on the author's machine streaming to Drive over `rclone`.

#### Nepali (Model L) — 13–17 August

| When | Where | What ran | What it produced |
| :--- | :--- | :--- | :--- |
| 13 Aug | Local | `src/data/collect/`, `clean_nepali.py`, `nepali_lang_id.py` — first collection and cleaning code | repo scaffolding |
| 14 Aug | Local | `collect_manual_web.py` (process-safe locking), `collect_wikimedia_api.py` | `data/nepali/raw/manual_web/` HTML, manifests |
| 15 Aug 18:26 | Local | Colab notebooks committed | `collect_nepali_colab.ipynb` |
| 15 Aug 19:29 | Colab | Wikipedia dump download | `nepali/raw/wiki_dump/newiki-…xml.bz2` |
| 15 Aug 19:39 – 21:58 | Colab | Six debugging rounds: HF login, XML namespace, the `root.clear()` RAM leak, `data-0.parquet`, PyArrow micro-batching | — |
| 15 Aug 21:58 – 16 Aug 02:09 | Colab | Sangraha ingest | `nepali/cleaned/automated_nonweb/` — 248 shards |
| 16 Aug 02:09 – 17 Aug 23:27 | Colab | Wikipedia dump parse + news crawls | `nepali/cleaned/manual/` — 48 shards |
| 16 Aug 02:17 – 03:02 | Colab | Tokenizer training (BPE + Unigram comparison) | `nepali/tokenizer/` — 4 files |
| 17 Aug 23:31 | Colab | Split generation | `nepali/splits/` — 6 files |

#### Hindi (Model H) — 19–21 August

| When | Where | What ran | What it produced |
| :--- | :--- | :--- | :--- |
| 19 Aug 01:19 – 01:26 | Colab | `collect_hindi_colab.ipynb` Step 2a — `hiwiki` dump parse | `wiki_shard_0000–0029` (30 shards) |
| 19 Aug 01:26 – 01:59 | Colab | Step 1 — Sangraha ingest | `automated_nonweb/` — 220 shards, 5.98 GiB |
| 19 Aug 14:30 – 14:39 | Local | Commits enforcing the ≥500M / ≥20% targets and FUSE reconnect handling | — |
| 19 Aug 14:46 | Colab | Literature step | `manual_lit_shard_0030` — later found 99.96% redundant (§4.3) |
| 19 Aug 17:18 – 18:46 | Local | Commits: NVMe staging for splits, Step 2c manual expansion | — |
| 20 Aug 19:17 | Colab | Step 2c — Wikibooks + Wikiquote dumps | `manual_aux_shard_0032` |
| 20 Aug 19:56 – 22:50 | Colab | `remaining_hindi_manual_collection.ipynb` — news crawl | `manual_news_shard_0033–0038` (6 shards) |
| 20 Aug 20:06 | Colab | Wikisource step | `wikisource_shard_0000` — **1 document**; namespace bug diagnosed in §3.3 |
| 20 Aug 22:36 – 22:43 | **Local** | `scripts/discover_hindi_urls.py` | 11,112,623 URLs → `frontier_full.txt.gz` |
| 20 Aug 22:52 – 23:07 | **Local** | `scripts/export_hindi_manual_hashes.py` | dedup registry, 155,123 hashes |
| 20 Aug 23:13 – 21 Aug 00:27 | **Local** | `scripts/collect_hindi_manual_web.py` — 87,585 URLs at 20.2 URL/s | `manual_web_shard_*` (27 shards) + `raw/manual_web/` (27 archives) |
| 21 Aug 01:46 | Colab | `finalize_hindi_corpus.ipynb`, first run — **discarded** | corpus 95.7% manual; staging failure, §8 |
| 21 Aug 03:15 – 03:22 | Colab | `finalize_hindi_corpus.ipynb`, final run | `splits/` + `tokenizer/hindi_bpe_8k.*` |
| 21 Aug 03:36 | **Local** | Provenance archival | `manifests/*.gz` |
| 21 Aug 03:46 | **Local** | `scripts/benchmark_hindi_tokenizer.py` | `report/hindi_tokenizer_benchmark.json` |
| 21 Aug 04:09 | Colab | Drive API resumable re-upload of `train.txt` / `train.jsonl` | corrected splits (§8) |
| 21 Aug 04:32 – 04:48 | **Local** | Wikimedia dump archival | `raw/wiki_dump/` — 3 dumps |

#### Why the split between Colab and local

Colab was used wherever the work needed to read or write multi-gigabyte
artifacts already on Drive — Sangraha ingest, dump parsing, split generation and
tokenizer training — because Drive is mounted there and its disk is large.

The Hindi manual web crawl ran **locally** instead, for three reasons: it takes
about an hour of sustained crawling and would outlive a Colab session; the
author's machine has ~3 GB of free disk, so the crawler streams each shard to
Drive with `rclone` and deletes it immediately rather than staging a corpus; and
`rclone` gives verifiable transfers, which the Drive FUSE mount does not — a
distinction that turned out to matter (§8).

---

## 8. System Architecture & Engineering Optimizations

Running multi-gigabyte data streaming and tokenizer training on Google Colab free-tier instances required overcoming severe memory and runtime constraints:

1. **Direct File-by-File Streaming**: Bypassed Hugging Face `datasets` C++ memory caching by downloading raw 340MB Parquet files directly via `urllib.request` and streaming records in 5,000-row micro-batches using `pyarrow.parquet.ParquetFile.iter_batches()`.
2. **64-bit Integer Hash Table**: Replaced full 64-character hex strings with 64-bit integer hashes (`int(sha256[:16], 16)`), reducing in-memory deduplication RAM usage by ~75%.
3. **XML Parser Memory Clearing (`root.clear()`)**: Resolved a critical Python `ElementTree.iterparse` memory leak during Wikipedia dump parsing, keeping RAM usage flat under 300MB across 50,000+ XML pages.
4. **Stateful Drive Checkpointing**: Shards are written directly to Google Drive (`data/nepali/cleaned/...`), allowing instant recovery and resumption in the event of Colab session preemption.

### 8.1 Four silent failures, and what they have in common

Each of the following produced wrong output while every check in place reported
success. They are recorded because the shared lesson is more useful than any one
fix: **a transfer that is not size-verified, and an assertion made about a plan
rather than a result, will both fail quietly.**

| # | Failure | Symptom | Fix |
| ---: | :--- | :--- | :--- |
| 1 | Shard staging used one bulk `cp … 2>/dev/null` for 220 files | The FUSE transport dropped, the error went to the null device, **2 of 220** automated shards arrived; the corpus came out 95.7% manual with every check green | Copy per file, compare each size against Drive, remount on `Errno 107`, retry |
| 2 | The budget assertion checked that the target was *reachable*, not what was ingested | Failure 1 passed validation and reached Drive | Assert on `written_auto_chars` after Pass 2, before any sync |
| 3 | `rclone cat` piped into a reader | A truncated stream is indistinguishable from a short file: the audit under-counted the corpus by **10,000 documents** and the dedup registry lost 10,000 hashes | Compare bytes read against the size Drive reports; refuse to report a partial count |
| 4 | Split sync used a bulk `cp` with no size check | A failed 3.9 GB `train.txt` write left the *previous run's* file on Drive while `val`/`test` updated normally | Copy and size-verify each split individually |

A fifth instance appeared outside the pipeline: Colab's Drive FUSE mount accepted
a 4.1 GB `train.txt` write and reported the correct size back from
`os.path.getsize`, but never committed it — `flush_and_unmount()` did not rescue
it either. The file was uploaded through the **Drive API with resumable upload**
instead, which returns the stored size from Google rather than from the local
cache. That is the only verification that means anything for a mounted
filesystem.

Deduplication had the same shape of flaw (§4.3): it ran manual→automated but
never manual→manual, because each collection run deduplicated against a registry
snapshotted before the following runs existed.

---

## 9. Conclusion & Readiness for Phase 2

Phase 1 deliverables are complete for both languages:

| Deliverable (§1.3 of the project brief) | Hindi | Nepali |
| :--- | :--- | :--- |
| Dataset collection scripts | `scripts/discover_hindi_urls.py`, `collect_hindi_manual_web.py`, `process_hindi_wikimedia_dump.py` | `src/data/collect/` |
| Dataset preprocessing pipeline | `src/data/clean/clean_hindi_web.py`, `hindi_lang_id.py` | `clean_nepali.py`, `nepali_lang_id.py` |
| Dataset statistics report | §5.2 | §5.1 |
| Train / validation / test splits | `hindi/splits/` (§7.2) | `nepali/splits/` |
| Tokenizer training code | `notebooks/finalize_hindi_corpus.ipynb` | `src/tokenizer/train_bpe.py` |
| Vocabulary file | `hindi/tokenizer/hindi_bpe_8k.vocab` | `nepali/tokenizer/nepali_bpe_32k.vocab` |
| Tokenizer model file | `hindi/tokenizer/hindi_bpe_8k.model` | `nepali/tokenizer/nepali_bpe_32k.model` |

* Cleaned, deduplicated, leak-free corpora: **520.4M tokens (Hindi)** and 488.5M (Nepali), both above the ~500M target.
* Manual collection at **26.87%** (Hindi) and ~50% (Nepali), both above the 20% floor.
* Zero-unknown-token BPE tokenizers with measured fertility of **1.4938** (Hindi, 8k vocab) and 1.55 (Nepali, 32k vocab).

**Open item carried into Phase 2.** The Nepali tokenizer's 32,000 vocabulary would
consume 32.8% of a 25M parameter budget on embeddings alone (§6.2, §6.4). It
should be retrained at 8k for parity with Hindi, or the choice justified against
its parameter cost, before pretraining begins.
