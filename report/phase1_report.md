# Phase 1 Report: Data Collection, Preprocessing, and Tokenizer Construction
**Course**: Language Models and Agents (CL3-410) — Monsoon 2026  
**Project**: Building and Analyzing Monolingual Transformer LMs in Two Indian Languages  
**Author**: 24KaratAu (Avnish Uba)


---

## 1. Executive Summary

Phase 1 establishes the foundational data and tokenization pipelines for two completely independent monolingual decoder-only Transformer language models:
* **Model H (Higher-Resource)**: **Hindi** (`hi`)
* **Model L (Lower-Resource)**: **Nepali** (`ne`)

Each language pipeline operates in complete isolation with dedicated raw data collection, custom text cleaning, deduplication, 90/5/5 document-level train/validation/test splits, and its own SentencePiece Byte-Pair Encoding (BPE) tokenizer trained from scratch. No corpus, vocabulary or tokenizer is shared between the two models.

**Headline results**

| | Hindi (Model H) | Nepali (Model L) |
| :--- | ---: | ---: |
| Active pretraining corpus | **520,409,408 tokens** | **632,711,308 tokens** |
| Manual share (floor is 20%) | **26.87%** | **21.20%** |
| Documents | 848,815 | 1,367,965 |
| Unique manual pool | 479,582,790 chars | 541,814,473 chars |
| Tokenizer vocabulary | 8,000 | 8,000 |
| Fertility (tokens/word) | **1.4938** | 1.5545 (sampled) |
| Characters per token | 3.484 | 4.127 |
| Unknown-token rate | **0.00%** | **0.00%** |

Both corpora clear the ~500M token target and the ≥20% manual-collection floor.
Reaching that point required correcting two silent pipeline failures per language
(§8.1) and, for Nepali, discovering that only 42.7% of its manual partition was
unique and collecting 340M further characters to compensate (§3.4, §4.3).

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

**Sitemap discovery.** The replacement (`scripts/discover_sitemap_urls.py`) reads
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

**Crawl results.** 75,002 documents / 161,000,865 clean characters across 27
shards were accepted from **87,585 attempted URLs (85.6% acceptance)** at
20.2 URL/s; 75,387 documents / 160,980,352 characters survive corpus-wide
deduplication. Rejections: 5,556 `robots.txt` disallow (entirely
`hindi.oneindia.com`, whose article paths are disallowed for all agents), 4,978
quality filter, 915 HTTP 410, 335 extraction failures, 222 HTTP 404, 132
timeouts/connection errors, 51 language-ID uncertain, and **1 document rejected
as Nepali or Sanskrit**. Figures are from
`data/hindi/manifests/manual_web_crawl/crawl_summary.json`. Deduplication ran against a registry of the 155,123 content hashes already
present in the manual partition.

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

### 3.4 Nepali Manual Collection: Remediation Crawl (21 August)

The audit described in §4.3 established that the Nepali manual partition was
only 42.7% unique, leaving **~43.1M unique manual tokens** and capping the whole
corpus at `5 x 43.1M = ~215M` tokens under the 20% floor — with 544M automated
Sangraha tokens sitting unusable in reserve. A second collection round was
therefore required, using the same sitemap machinery built for Hindi
(`scripts/discover_sitemap_urls.py`, `scripts/collect_manual_web_sitemap.py
--language nepali`).

**Nepali is a materially harder crawl than Hindi.** Probing 38 domains found
that several of the largest outlets — `ekantipur.com`, `onlinekhabar.com`,
`ratopati.com`, `himalkhabar.com`, `nagariknews.nagariknetwork.com`,
`annapurnapost.com` — publish **no sitemap at all**, neither via `robots.txt`
nor at the conventional path, so the method simply cannot reach them. Discovery
returned **651,660 URLs concentrated in 8 substantial hosts**, against 11.1M
across 18 hosts for Hindi.

The first discovery pass returned 309,227 URLs from only four hosts, three of
them business/finance publications. Rather than crawl a corpus that was
effectively three finance outlets, the domain list was widened before crawling;
the second pass added `makalukhabar.com` (167k URLs), `kathmandupati.com`
(112k), `nepalpatra.com` and `nepalnews.com`, all general news. This both fixed
the register skew and raised throughput from 9.1 to 24.6 URL/s, since the
crawler caps concurrency per host.

**Results.** 202,769 documents / 340,215,209 clean characters accepted from
227,699 attempted URLs (**89.1% acceptance**) across 70 shards, with matching
gzipped raw HTML. A 9,000-document sample contained **zero duplicates**, and no
single host exceeds 20.8% of the crawl:

| Host | Accepted | Share | | Host | Accepted | Share |
| :--- | ---: | ---: | --- | :--- | ---: | ---: |
| `makalukhabar.com` | 42,184 | 20.8% | | `arthasarokar.com` | 30,248 | 14.9% |
| `kathmandupati.com` | 39,639 | 19.5% | | `deshsanchar.com` | 12,543 | 6.2% |
| `bizmandu.com` | 34,862 | 17.2% | | `nepalnews.com` | 8,868 | 4.4% |
| `clickmandu.com` | 34,090 | 16.8% | | 15 further hosts | 335 | 0.2% |

Rejections: 11,485 HTTP 404, 6,725 language-ID uncertain, 2,870 quality filter,
2,415 HTTP 429, 1,009 duplicate, 162 HTTP 500, 80 connection errors, and **19
documents rejected as Hindi** — the cross-lingual guard doing precisely the job
it exists for, since Hindi is Model H's language and the two corpora must remain
disjoint. Figures are from
`data/nepali/manifests/manual_web_crawl/crawl_summary.json`, which aggregates
every attempt including the two aborted runs, and so exceeds any single run's
console summary.

Two crawler changes were needed for Nepali. Smaller Nepali portals rate-limit far
sooner than the large Hindi dailies (7.3% of pilot requests returned HTTP 429),
so 429/503 now trigger a `Retry-After`-aware backoff instead of discarding the
page. And the line-level cleaner was parameterised by marker vocabulary rather
than forked, so `clean_nepali_web.py` supplies Nepali boilerplate and teaser
strings (`यो पनि पढ्नुहोस्`, `सेयर गर्नुहोस्`, `प्रकाशित मिति`) over the same
language-neutral machinery.

**Outcome.** Unique manual rose from ~43.1M to **~110.0M tokens**
(549,501,090 characters), lifting the corpus ceiling from ~215M to **~550M
tokens** and making a ~520M corpus at ~21% manual reachable.

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

#### Verifying the corpora share no documents

The brief requires that the two corpora **share no documents and are never
concatenated**. Because both draw on the same upstream projects — AI4Bharat
Sangraha and Wikimedia, each in a different language configuration — this was
verified by content-hash intersection rather than assumed:

| Comparison | Hindi hashes | Nepali hashes | Shared |
| :--- | ---: | ---: | ---: |
| Full manual registries | 155,123 | 100,929 | **0** |
| Manual URLs | 14,789 | 197,695 | **0** |
| Automated — Sangraha `verified/hin` vs `verified/nep` | 10,000 | 10,000 | **0** |
| Sitemap web crawls | 3,000 | 3,000 | **0** |
| Wikipedia dumps — `hiwiki` vs `newiki` | 5,000 | 5,000 | **0** |

The first two rows are exhaustive over the manual partitions; the remaining
three are sampled. Independence is structural rather than accidental — separate
dataset configurations, separate dumps, disjoint crawl domain lists — and the
language identifier enforces it at the document level, having rejected 17 Hindi
pages found on Nepali news sites during the 2026-08-21 crawl (§3.4).

#### The same failure at far greater scale (Nepali)

Auditing Nepali the same way found the problem was not marginal there but
dominant. Streaming all 48 pre-existing manual shards and counting only
first-occurrence documents:

| Source | Unique documents | Unique characters |
| :--- | :--- | :--- |
| Wikipedia dump (`wiki_shard_*`) | 27,946 / 27,946 — **100%** | 42,864,542 / 42,864,542 — **100%** |
| Legacy news crawl (`manual_news_*`) | 76,813 / 208,975 — **36.8%** | 166,421,339 / 447,189,127 — **37.2%** |
| **Total manual** | **104,759 / 236,921 — 44.2%** | **209,285,881 / 490,053,669 — 42.7%** |

**Under 43% of the Nepali manual partition was unique**, and every duplicate came
from the legacy news crawl; the Wikipedia pipeline was clean. The mechanism is
visible in a direct comparison of two shards:

| | Shared content hashes | Shared URLs |
| :--- | ---: | ---: |
| `manual_news_shard_0030` ∩ `_0040` | 1,718 (34.4%) | **0** |
| `manual_news_shard_0020` ∩ `_0030` | 944 (18.9%) | **0** |

Identical article text under **completely disjoint URL sets**. Nepali outlets
republish agency copy, and the same story is served from several category paths,
so a crawler deduplicating on URL — as the legacy collector did — sees nothing
wrong. Only content hashing catches it.

The consequence was material: unique manual was ~43.1M tokens rather than the
~96M a document count implied, capping the entire Nepali corpus at `5 x 43.1M =
~215M` tokens under the 20% floor. This is what motivated the 2026-08-21 crawl
(§3.4), which deduplicates on content hash both within the run and against a
registry of all 100,929 pre-existing manual hashes.

**Generalisable lesson.** URL-level deduplication is not deduplication. It is
cheap and catches re-fetches, but it cannot see syndication, and in news corpora
syndication is the dominant duplicate mechanism. Content hashing must run at
collection time *and* again over the assembled partition, because collection-time
registries are only ever snapshots of what existed when the run started.

---

## 5. Dataset Statistics & Split Stratification

### 5.1 Dataset Scale (Nepali Corpus — Model L)

All figures are **post-deduplication** and measured directly from the shards on
Drive with `nepali_bpe_32k`. They supersede an earlier version of this report
which counted duplicate documents and consequently overstated the Nepali corpus
by roughly 4.8x; §4.3 documents how the duplication arose and how it was found.

| Partition / Source Category | Shards | Unique Documents | Unique Clean Characters | Chars/Token | Tokens |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Manual — Wikipedia dump (`wiki_shard_*`) | 6 | 27,946 | 42,864,542 | 4.853 | ~8,832,000 |
| Manual — Web crawl, sitemap (`manual_web_shard_*`) | 66 | 202,769 | 340,215,209 | 5.091 | ~66,827,000 |
| Manual — News crawl, legacy (`manual_news_*`) | 42 | 76,813 | 166,421,339 | 4.853 | ~34,292,000 |
| **Subtotal: Total Manual (unique)** | **114** | **307,528** | **549,501,090** | **4.99** | **~109,951,000** |
| Automated — Sangraha Verified (`nonweb_shard_*`) | 248 | ~1,090,000 | ~2,445,256,140 | 4.493 | ~544,293,000 |
| **Total Clean Reserve Pool on Drive** | **362** | **~1,397,500** | **~2,994,757,230** | — | **~654,244,000** |

**Deduplication impact.** The 42 legacy news shards contained 208,975 documents
/ 447,189,127 characters, of which only **76,813 documents / 166,421,339
characters (37.2%) are unique**. The 6 Wikipedia dump shards are **100% unique**
(27,946/27,946). The duplication is therefore entirely an artefact of the legacy
news crawl, not of the Wikipedia pipeline — see §4.3.

**Why the 2026-08-21 crawl was necessary.** Before it, unique manual stood at
~43.1M tokens, capping the corpus at `5 x 43.1M = ~215M` tokens under the 20%
floor — far below the ~500M target, with 544M automated tokens unusable in
reserve. The crawl added 340,215,209 characters (~66.8M tokens), raising unique
manual to ~110.0M and the ceiling to **~550M tokens**.

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

Produced by `notebooks/finalize_nepali_corpus.ipynb`, which applies the same
verified staging, intra-manual deduplication, budget derivation and ingest
assertions as the Hindi notebook. Deduplication removed **132,181 documents /
280,791,280 characters** from the manual partition before the budget was derived
from it (§4.3). Token counts are measured with the final `nepali_bpe_8k`.

| Split Name | Percentage | Document Count | Clean Characters | BPE Tokens (`8k` Vocab) |
| :--- | :---: | :---: | :---: | :---: |
| **Train** (`train.txt`) | 90.0% | 1,231,136 | 2,350,842,501 | **569,663,061** |
| **Validation** (`val.txt`) | 5.0% | 68,583 | 130,705,992 | **31,697,868** |
| **Test** (`test.txt`) | 5.0% | 68,246 | 129,538,486 | **31,350,379** |
| **Total Active Pretraining Corpus** | **100.0%** | **1,367,965** | **2,611,086,979** | **632,711,308 (~632.7M tokens)** |

| Collection Partition | Unique Clean Characters | Chars/Token | Tokens | Share |
| :--- | ---: | ---: | ---: | ---: |
| Manual (100% of unique manual ingested) | 541,814,473 | 4.014 | ~134,987,000 | **21.20%** |
| Automated (budget-limited) | 2,069,272,506 | 4.123 | ~501,880,000 | 78.80% |

**On the 632.7M rather than 520M.** The automated budget was derived from
characters-per-token measured on the *pre-existing* `nepali_bpe_32k` (5.022), and
the corpus was then encoded with the *newly trained* 8k model (4.127). The same
2.61B characters therefore produced 632.7M tokens instead of the projected 520M —
overshooting the ~500M target rather than undershooting it. Both requirements are
met, so no re-split was run; §8.1 records the lesson for pipelines that both
consume and produce a tokenizer.

**On the manual share.** Cell 3 reported 20.75% using the 32k ratios, where manual
and automated were near-identical (5.022 vs 5.021). Re-measured directly against
`nepali_bpe_8k`, manual compresses better than Sangraha (4.014 vs 4.123) and so
contributes proportionally more tokens: the true share is **21.20%**.

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
| `split_digits` | `True` | Each **ASCII** digit becomes its own token, for arithmetic in Phase 3 — see the caveat below |
| Special tokens | `<pad>=0, <unk>=1, <bos>=2, <eos>=3` | Fixed IDs shared by both models |

> **Caveat on `split_digits`, and why it does not matter here.** The flag splits
> **ASCII** digits only; Devanagari numerals (`०–९`, U+0966–U+096F) are merged by
> BPE like any other character sequence:
>
> | Input | Tokenized as |
> | :--- | :--- |
> | `2026` | `▁ 2 0 2 6` — one token per digit |
> | `२०२६` | `▁२०२ ६` (Nepali 8k) · `▁२० २ ६` (Hindi 8k) |
>
> The consequence depends entirely on which numeral system each corpus actually
> uses, so we measured it. Over ~12M characters of held-out text per language:
>
> | Corpus | ASCII `0–9` | Devanagari `०–९` |
> | :--- | ---: | ---: |
> | Hindi | **94.8%** | 5.2% |
> | Nepali | 0.1% | **99.9%** |
>
> The two corpora have opposite conventions, and each is well served by the
> tokenizer it got. Hindi text uses ASCII digits, which `split_digits` separates
> per digit as intended. Nepali text uses Devanagari numerals, and **every value
> from ० to १०० encodes as a single token** (101 of 101) — for the comparative
> reasoning Phase 3 requires, an atomic token per quantity is at least as useful
> as per-digit splitting. Only four-digit-plus values fragment, which comparisons
> over ages, heights and prices do not reach.
>
> The operative guidance for Phase 3 is therefore: **write each reasoning dataset
> in its own corpus's numeral convention** — ASCII for Hindi, Devanagari for
> Nepali — and keep quantities to three digits. Converting Nepali to ASCII would
> be both unnatural phrasing and badly out of distribution: ASCII digits occur
> 152 times in 11.7M characters of Nepali.

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

#### 6.2.4 Choosing the vocabulary size: the selection rule

There is no free-standing "optimal" vocabulary. The real objective is *minimise
validation loss subject to a ~25M parameter budget*, and that cannot be measured
until Phase 2. What follows is therefore a rule built from three things that
**can** be measured now, stated explicitly so the choice is auditable rather
than asserted.

**(a) Marginal return on parameters.** Embedding cost is `V x d_model`, so every
doubling of the vocabulary doubles the spend. Measured on held-out Nepali text:

| Step | Extra embedding params | Fertility gained | Gain per 1M params |
| :--- | ---: | ---: | ---: |
| 4k → 8k | 1,024,000 | 0.2638 | **0.2576** |
| 8k → 16k | 2,048,000 | 0.1710 | 0.0835 |
| 16k → 32k | 4,096,000 | 0.1048 | **0.0256** |

Each doubling costs twice as much and returns roughly a third as much per
parameter; 16k→32k is **ten times less parameter-efficient** than 4k→8k. Hindi
shows the same shape.

**(b) The choice is coupled to `d_model`, which Phase 2 has not yet fixed.**
Embedding share of a 25M budget:

| `d_model` | V=8k | V=16k | V=32k |
| ---: | ---: | ---: | ---: |
| 256 | 8.2% | 16.4% | 32.8% |
| 384 | 12.3% | 24.6% | 49.2% |
| 512 | **16.4%** | 32.8% | **65.5%** |

At `d_model = 512` — an ordinary width for a 25M decoder — a 32k vocabulary
consumes **65.5% of the entire budget**, leaving roughly 2.7 transformer blocks.
The model would be mostly a lookup table.

**(c) The selection rule.** *Choose the largest vocabulary whose embedding table
stays affordable across every `d_model` still under consideration.* Only
**8,000** satisfies this: it costs between 8.2% and 16.4% depending on width.
16k is defensible only by committing now to `d_model <= 256`; 32k only at 256
*and* accepting a third of the model on embeddings. Selecting the option that is
robust to a decision not yet taken is a stronger position than optimising
against an architecture that may still change.

Supporting, but not decisive: the course Q&A advises *"a vocab size of 5 to
10K"*, and using 8,000 for both models means Phase 3 cross-language comparisons
differ by language and data rather than by tokenizer capacity.

**What this costs, stated plainly.** A 32k vocabulary fits **21.6% more Nepali
text into the same context window** than 8k (fertility 1.2787 vs 1.5545), and
halves the broken-grapheme rate (1.79% vs 4.94%). Those are real losses, not
rounding. The rule above trades them for model depth.

**What would settle it properly.** Train Model L at 8k and at 16k and compare
*validation loss and bits-per-byte* — the metric that actually matters — rather
than fertility, which is only a proxy. That is a Phase 2 ablation, and it is
named here rather than quietly skipped.

#### 6.2.5 Empirical sweep, both languages

For each language we fitted BPE at four vocabulary sizes plus a Unigram LM at 8k
on an 80 MB sample of that language's **training** split, and evaluated every one
on held-out documents from its **validation** split. No model is scored on text
it was fitted on.

**Hindi** — 59,713 held-out documents (12.2M characters):

| Model | Vocab | Fertility | Chars/token | Unknown | Broken graphemes | Coverage |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| BPE | 4,000 | 1.7855 | 2.9041 | 0.00% | 11.229% | 94.88% |
| **BPE** | **8,000** | **1.5873** | **3.2666** | **0.00%** | 7.967% | 94.31% |
| Unigram | 8,000 | 1.5613 | 3.3210 | 0.00% | 6.073% | 94.30% |
| BPE | 16,000 | 1.4509 | 3.5738 | 0.00% | 5.388% | 91.23% |
| BPE | 32,000 | 1.3587 | 3.8163 | 0.00% | 3.502% | **82.39%** |

**Nepali** — 70,196 held-out documents:

| Model | Vocab | Fertility | Chars/token | Unknown | Broken graphemes | Coverage |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| BPE | 4,000 | 1.8183 | 3.6415 | 0.00% | 7.458% | 96.12% |
| **BPE** | **8,000** | **1.5545** | **4.2594** | **0.00%** | 4.942% | **97.45%** |
| Unigram | 8,000 | 1.5187 | 4.3598 | 0.00% | 5.034% | 98.06% |
| BPE | 16,000 | 1.3835 | 4.7857 | 0.00% | 3.070% | 97.24% |
| BPE | 32,000 | 1.2787 | 5.1783 | 0.00% | 1.791% | 93.93% |

Findings:

1. **Unknown-token rate is exactly 0.00% at every size in both languages**,
   confirming `byte_fallback` eliminates out-of-vocabulary failure. Vocabulary
   size therefore never has to be inflated defensively to cover rare conjuncts,
   nuktas or stray Latin — it can be chosen purely on the parameter trade-off.
2. **The two languages behave differently at 32k, and this matters.** Hindi
   coverage collapses to 82.39%: about 5,600 embedding rows (~1.4M parameters)
   are never observed on held-out text, so a 32k Hindi vocabulary is simply
   wasteful. **Nepali holds 93.93%.** Its agglutinative morphology — bound case
   markers `-लाई`, `-बाट`, `-को`, plural `-हरू`, multi-part verb forms — produces
   enough distinct high-frequency subword types to fill a large vocabulary.
3. **So the argument that settles Hindi does not transfer to Nepali.** For
   Nepali, 32k is not wasteful, only expensive; the case for 8k rests entirely on
   the parameter rule in §6.2.4, not on dead weight. This is stated explicitly
   because using Hindi's reasoning for Nepali would be wrong.
4. **Nepali compresses better than Hindi at every size** (4.2594 vs 3.2666
   chars/token at 8k), for the same morphological reason: longer reusable stems.

**Selected for both models: BPE, 8,000-token vocabulary, byte fallback.**
Vocabulary size follows the rule in §6.2.4; the algorithm follows the procedural
grounds in §6.2.2, which flags it as the decision most worth revisiting.

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

### 6.4 Nepali (Model L) — `nepali_bpe_8k`

Nepali's tokenizer is being retrained at 8,000 to match the rule in §6.2.4. The
figures below are for the **superseded `nepali_bpe_32k`**, measured on 70,196
held-out validation documents; they are retained because they are what the 32k
decision was originally made on, and because the comparison is informative.

| Statistic | `nepali_bpe_32k` (superseded) | `nepali_bpe_8k` (selected)¹ |
| :--- | ---: | ---: |
| Vocabulary size | 32,000 | 8,000 |
| Embedding parameters (`d_model=512`) | 16,384,000 (**65.5%** of 25M) | 4,096,000 (16.4%) |
| Fertility (tokens/word) | 1.3170 | 1.5545 |
| Characters per token | 5.0274 | 4.2594 |
| Unknown-token rate | 0.00% | 0.00% |
| Broken grapheme rate | 1.609% | 4.942% |
| Vocabulary coverage | 91.92% | 97.45% |
| Shannon entropy | 11.7805 bits | — |
| Rényi entropy (α=2) | 8.3181 bits | — |

¹ 8k figures are from the §6.2.5 sweep, fitted on an 80 MB training sample. As
with Hindi, the production model trained on the full split should improve on
them — Hindi's production 8k reached fertility 1.4938 against 1.5873 sampled.

**Why 32k was not kept.** It is the better tokenizer in isolation: fertility
1.3170 against 1.5545, broken graphemes 1.609% against 4.942%, and — unlike
Hindi — its coverage stays healthy at 91.92%, so it is not padded with unused
entries. The objection is purely budgetary. At `d_model = 512` its embedding
table alone is **65.5% of a 25M model**, which cannot be reconciled with
building a transformer underneath it. §6.2.4 sets out the rule; this is the case
where the rule costs something real, and the cost is recorded rather than
minimised.

**Measured statistics for the shipped `nepali_bpe_8k`**, on 70,196 held-out
validation documents:

| Statistic | Value |
| :--- | ---: |
| Vocabulary size | **8,000** |
| Fertility (tokens/word) | **1.5566** |
| Characters per token | **4.2537** |
| Unknown tokens | **0** (rate **0.00%**) |
| Broken grapheme rate | 4.462% |
| Vocabulary types observed | 7,759 of 8,000 (**96.99%**) |
| Vocabulary types seen exactly once | 63 |
| Shannon entropy | 11.1083 bits |
| Rényi entropy (α=2) | 8.6409 bits |

**Token-frequency statistics.**

| Cumulative coverage | Share of all tokens |
| :--- | ---: |
| Top 100 pieces | 28.38% |
| Top 500 pieces | 51.07% |
| Top 1,000 pieces | 64.34% |
| Top 2,000 pieces | 78.22% |
| Top 4,000 pieces | 91.59% |

The distribution is noticeably flatter than Hindi's, where the top 100 pieces
already cover 40.34% against Nepali's 28.38% — consistent with Nepali spreading
probability mass over a larger inventory of inflected forms.

| Rank | Piece | Gloss | Count | Share |
| ---: | :--- | :--- | ---: | ---: |
| 1 | `।` | danda (full stop) | 79,610 | 2.924% |
| 2 | `,` | comma | 48,780 | 1.791% |
| 3 | `▁र` | *ra* ("and") | 37,268 | 1.369% |
| 4 | `▁छ` | *cha* ("is") | 33,318 | 1.224% |
| 5 | `▁।` | danda after space | 28,073 | 1.031% |
| 6 | **`को`** | genitive *-ko* | 26,709 | 0.981% |
| 7 | **`मा`** | locative *-ma* | 23,070 | 0.847% |
| 8 | **`का`** | genitive *-ka* | 21,174 | 0.778% |
| 9 | **`ले`** | ergative *-le* | 19,163 | 0.704% |
| 10 | **`लाई`** | accusative/dative *-lai* | 16,476 | 0.605% |
| 11 | `▁पनि` | *pani* ("also") | 16,455 | 0.604% |
| 12 | `▁छन्` | *chan* ("are") | 13,544 | 0.497% |
| 13 | `▁भएको` | *bhaeko* ("having become") | 10,400 | 0.382% |
| 14 | `▁गरेको` | *gareko* ("having done") | 9,722 | 0.357% |
| 15 | `▁भने` | *bhane* ("said/if") | 9,695 | 0.356% |

**This table independently confirms the morphological claim in §2.2.** Ranks 6–10
— `को`, `मा`, `का`, `ले`, `लाई` — carry **no `▁` word-boundary marker**, meaning
BPE learned them as *suffixes bound to a preceding stem*. The corresponding Hindi
entries in §6.3 are `▁के`, `▁में`, `▁की`, `▁को`, `▁से` — all **with** `▁`, i.e.
free-standing words.

Neither tokenizer was told anything about morphology. Trained independently on
separate corpora, they each recovered the distinction §2.2 asserts on linguistic
grounds: Hindi postpositions are separate words, Nepali case markers are bound
morphemes. It is the clearest evidence in Phase 1 that the two pipelines really
are modelling two different languages rather than one script.

**Nepali compresses better than Hindi at equal vocabulary** — 4.2537 vs 3.4712
chars/token, both at 8k. Nepali's bound morphology gives BPE longer, more reusable stems.

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

Produced with the shipped `nepali_bpe_8k`, shown against the superseded 32k model
so the cost of the vocabulary decision (§6.2.4) is visible rather than asserted.
`▁` marks a word boundary.

```text
नेपाल एक सुन्दर, सार्वभौमसत्ता सम्पन्न दक्षिण एशियाली राष्ट्र हो।
   8k (16 tok, fertility 1.78): ▁नेपाल ▁एक ▁सुन्दर , ▁सार्व भौ म स त्ता ▁सम्पन्न ▁दक्षिण ▁एश ियाली ▁राष्ट्र ▁हो ।
  32k (11 tok, fertility 1.22): ▁नेपाल ▁एक ▁सुन्दर , ▁सार्वभौमसत्ता ▁सम्पन्न ▁दक्षिण ▁एशियाली ▁राष्ट्र ▁हो ।
  This is the worst case for 8k. सार्वभौमसत्ता ("sovereignty") shatters into five
  meaningless fragments, and एशियाली breaks mid-syllable at ▁एश|ियाली — the
  broken-grapheme failure the 4.942% rate in §6.4 is counting.

काठमाडौं उपत्यकाभित्र विभिन्न ऐतिहासिक दृष्टिकोणले महत्त्वपूर्ण स्थलहरू छन्।
   8k (12 tok, fertility 1.50): ▁काठमाडौं ▁उपत्यका भित्र ▁विभिन्न ▁ऐतिहासिक ▁दृष्टिकोण ले ▁महत्त्वपूर्ण ▁स्थल हरू ▁छन् ।
  32k (10 tok, fertility 1.25): ▁काठमाडौं ▁उपत्यकाभित्र ▁विभिन्न ▁ऐतिहासिक ▁दृष्टिकोणले ▁महत्त्वपूर्ण ▁स्थल हरू ▁छन् ।
  Here 8k's extra splits fall on real morpheme boundaries: उपत्यका|भित्र
  ("valley"+"inside") and दृष्टिकोण|ले ("viewpoint"+ergative -le). Both models
  agree on स्थल|हरू (stem + plural -haru).

विश्वविद्यालयमा विद्यार्थीहरूले प्राविधिक तथा वैज्ञानिक अनुसन्धान गरिरहेका छन्।
   8k (11 tok, fertility 1.38): ▁विश्वविद्यालय मा ▁विद्यार्थी हरूले ▁प्राविधिक ▁तथा ▁वैज्ञानिक ▁अनुसन्धान ▁गरिरहेका ▁छन् ।
  32k ( 9 tok, fertility 1.12): ▁विश्वविद्यालयमा ▁विद्यार्थीहरूले ▁प्राविधिक ▁तथा ▁वैज्ञानिक ▁अनुसन्धान ▁गरिरहेका ▁छन् ।
  8k separates the locative -मा and the plural-ergative -हरूले from their stems.
  For an agglutinative language this is arguably the more useful segmentation:
  the model sees विश्वविद्यालय and विद्यार्थी as reusable stems rather than
  memorising every inflected surface form.

प्रधानमन्त्रीले २०२६ मा अर्थतन्त्र १५ प्रतिशतले बढेको बताउनुभयो।
   8k (10 tok, fertility 1.25): ▁प्रधानमन्त्रीले ▁२०२ ६ ▁मा ▁अर्थतन्त्र ▁१५ ▁प्रतिशतले ▁बढेको ▁बताउनुभयो ।
  32k ( 9 tok, fertility 1.12): ▁प्रधानमन्त्रीले ▁२०२६ ▁मा ▁अर्थतन्त्र ▁१५ ▁प्रतिशतले ▁बढेको ▁बताउनुभयो ।
  Note ▁२०२|६ — the Devanagari year is split arbitrarily rather than per digit,
  because split_digits applies to ASCII only (§6.1).
```

**Reading these honestly.** 8k is measurably worse per-token than 32k on all four
samples, and one case (सार्वभौमसत्ता) is genuinely poor. The compensating
observations are that several of 8k's extra splits are morphologically principled
rather than arbitrary, and that the parameter budget freed — 12.3 percentage
points of a 25M model at `d_model = 512` — buys transformer depth that no
tokenizer improvement can substitute for. §6.2.4 sets out that trade; these
examples are what it costs in practice.

### 6.7 Reproducing these numbers

```bash
python scripts/benchmark_tokenizer.py \
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

| # | Source | What it is | How obtained | Collection type | Lands in |
| ---: | :--- | :--- | :--- | :--- | :--- |
| 1 | `ai4bharat/sangraha` (`verified/nep`) | Human-verified Indic corpus | Same direct-parquet streaming path as Hindi | **Automated** | `nepali/cleaned/automated_nonweb/` (248 shards) |
| 2 | Nepali Wikipedia | Full article dump | `dumps.wikimedia.org/newiki/latest/newiki-latest-pages-articles.xml.bz2`, parsed in-house | **Manual** | `nepali/cleaned/manual/wiki_shard_0000–0005.jsonl` |
| 3 | Nepali news portals (legacy crawl) | *Ratopati*, *Setopati*, *Gorkhapatra*, *OnlineKhabar* | `src/data/collect/collect_manual_web.py`, URL-seeded | **Manual** | `nepali/cleaned/manual/manual_news_*.jsonl` (42 shards) — only 37.2% unique, see §4.3 |
| 4 | 22 Nepali news portals (sitemap crawl, 2026-08-21) | Live article text | Sitemaps discovered from `robots.txt`; largest contributors `makalukhabar.com`, `kathmandupati.com`, `bizmandu.com`, `clickmandu.com`, `arthasarokar.com`, `deshsanchar.com`, `nepalnews.com` | **Manual** | `nepali/cleaned/manual/manual_web_shard_*.jsonl` (66 shards) |

**A note on Nepali sitemap availability.** Nepali was a materially harder crawl
than Hindi. Several of the largest outlets — `ekantipur.com`,
`onlinekhabar.com`, `ratopati.com`, `himalkhabar.com`,
`nagariknews.nagariknetwork.com` and `annapurnapost.com` — publish **no sitemap
at all**, in `robots.txt` or at the conventional path, so the sitemap method
cannot reach them. Discovery across 38 probed domains yielded 651,660 URLs
concentrated in 8 hosts, against 11.1M across 18 hosts for Hindi. The resulting
corpus therefore leans toward business and general-news registers from a smaller
set of publishers, which is recorded here as a limitation rather than smoothed
over.

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
| `nepali/cleaned/manual/` | 118 | 2.217 GiB | Wikipedia dump + legacy news + sitemap crawl shards |
| `nepali/raw/wiki_dump/` | 1 | 0.051 GiB | `newiki-latest-pages-articles.xml.bz2` |
| `nepali/raw/manual_web/` | 73 | 2.193 GiB | Raw crawled HTML, gzipped per shard |
| `nepali/splits/` | 6 | 2.659 GiB | `train/val/test` × `.txt` and `.jsonl` |
| `nepali/tokenizer/` | 4 | 0.004 GiB | `nepali_bpe_32k` model/vocab (+ Unigram); `nepali_bpe_8k` pending retrain (§6.4) |

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
| URL discovery | `scripts/discover_sitemap_urls.py` | `frontier*.txt` |
| Dedup registry | `scripts/export_manual_hashes.py` | `existing_manual_hashes.txt` |
| Web crawl | `scripts/collect_manual_web_sitemap.py --language {hindi,nepali}` | `cleaned/manual/manual_web_shard_*`, `raw/manual_web/` |
| Cleaning | `src/data/clean/clean_hindi_web.py` / `clean_nepali_web.py`, `hindi_lang_id.py` / `nepali_lang_id.py` | applied inline during collection |
| Accounting | `scripts/audit_hindi_corpus.py` | per-source token tables (§5.2) |
| Split + tokenizer | `notebooks/finalize_hindi_corpus.ipynb` / `notebooks/finalize_nepali_corpus.ipynb` | `splits/`, `tokenizer/`, `{hindi,nepali}_corpus_summary.json` |
| Tokenizer benchmark | `scripts/benchmark_tokenizer.py --language {hindi,nepali}` | `report/{hindi,nepali}_tokenizer_benchmark.json` |

### 7.5 Execution timeline — what ran where, and when

> **Full per-stage ledger**: [`report/collection_ledger.md`](collection_ledger.md),
> generated from [`collection_ledger.json`](collection_ledger.json). It records all
> 20 collection stages with the commit hash, the exact script or notebook cell, the
> source, the shard filenames produced, and the measured document/character counts
> for each. The summary below is the readable overview; the ledger is the record.

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
| 20 Aug 22:36 – 22:43 | **Local** | `scripts/discover_sitemap_urls.py` | 11,112,623 URLs → `frontier_full.txt.gz` |
| 20 Aug 22:52 – 23:07 | **Local** | `scripts/export_manual_hashes.py` | dedup registry, 155,123 hashes |
| 20 Aug 23:13 – 21 Aug 00:27 | **Local** | `scripts/collect_manual_web_sitemap.py` — 87,585 URLs at 20.2 URL/s | `manual_web_shard_*` (27 shards) + `raw/manual_web/` (27 archives) |
| 21 Aug 01:46 | Colab | `finalize_hindi_corpus.ipynb`, first run — **discarded** | corpus 95.7% manual; staging failure, §8 |
| 21 Aug 03:15 – 03:22 | Colab | `finalize_hindi_corpus.ipynb`, final run | `splits/` + `tokenizer/hindi_bpe_8k.*` |
| 21 Aug 03:36 | **Local** | Provenance archival | `manifests/*.gz` |
| 21 Aug 03:46 | **Local** | `scripts/benchmark_tokenizer.py` | `report/hindi_tokenizer_benchmark.json` |
| 21 Aug 04:09 | Colab | Drive API resumable re-upload of `train.txt` / `train.jsonl` | corrected splits (§8) |
| 21 Aug 04:32 – 04:48 | **Local** | Wikimedia dump archival | `raw/wiki_dump/` — 3 dumps |

#### Nepali (Model L) — 21 August, remediation

An audit on 21 August found the Nepali corpus was not what the report claimed
(§4.3, §5.1). The following ran to correct it.

| When | Where | What ran | What it produced |
| :--- | :--- | :--- | :--- |
| 21 Aug 05:17 – 05:30 | **Local** | Exact measurement of `val.txt`; sampling of the cleaned pool | Split figures inflated ~4.8x; pool ~640M tokens vs ~102.7M in the splits |
| 21 Aug 05:30 – 05:45 | **Local** | `scripts/export_manual_hashes.py` over 48 manual shards | 100,929 unique hashes from 233,091 docs — 43.3% unique |
| 21 Aug 05:32 – 05:36 | **Local** | `scripts/discover_sitemap_urls.py`, first pass | 309,227 URLs, 4 hosts (business-heavy) |
| 21 Aug 05:50 – 05:55 | **Local** | Discovery, expanded to 38 probed domains | 651,660 URLs over 8 substantial hosts |
| 21 Aug 06:13 – 10:06 | **Local** | `scripts/collect_manual_web_sitemap.py --language nepali` | 202,769 docs / 340,215,209 chars (66 shards + raw archives) |
| 21 Aug 10:06 – 12:20 | **Local** | Unique-character measurement over all manual shards | Unique manual 209,285,881 chars pre-crawl; Wikipedia 100% clean, news 37.2% |
| 21 Aug 12:30 – 13:00 | **Local** | `scripts/benchmark_tokenizer.py --language nepali` | `report/nepali_tokenizer_benchmark.json`, vocabulary sweep |
| pending | Colab | `notebooks/finalize_nepali_corpus.ipynb` | Re-split with intra-manual dedup + `nepali_bpe_8k` |

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

### 7.6 Reproducing every measured claim

No figure in this report rests on an uncommitted script. Each measurement below
maps to the command that regenerates it. Anything streamed from Drive is
byte-verified against the size Drive reports, so a truncated read fails loudly
rather than silently under-reporting (§8.1, failure 3).

| Claim (§) | Command |
| :--- | :--- |
| Per-source token accounting (§5.1, §5.2) | `python scripts/audit_hindi_corpus.py` |
| Manual uniqueness by source — 42.7% Nepali, Wikipedia 100%, news 37.2% (§4.3, §5.1) | `python scripts/verify_corpus_claims.py uniqueness --language nepali --tokenizer tokenizers/nepali/nepali_bpe_8k.model` |
| Same text under different URLs — 34.4% shared hashes, 0 shared URLs (§4.3) | `python scripts/verify_corpus_claims.py shard-overlap --language nepali` |
| The two corpora share no documents (§4.3) | `python scripts/verify_corpus_claims.py cross-corpus` |
| Unicode NFC is a verified no-op (§4.1) | `python scripts/verify_corpus_claims.py nfc --language hindi` |
| Numeral convention and 0–100 encoding (§6.1) | `python scripts/verify_corpus_claims.py numerals --language nepali --tokenizer tokenizers/nepali/nepali_bpe_8k.model` |
| Manual token share under the shipped tokenizer (§5.3) | `python scripts/verify_corpus_claims.py token-share --language nepali --tokenizer tokenizers/nepali/nepali_bpe_8k.model --manual-chars 541814473 --auto-chars 2069272506 --manual-shards … --auto-shards …` |
| Which outlets publish crawlable sitemaps (§3.3, §3.4) | `python scripts/probe_sitemap_availability.py --language nepali` |
| Fertility, coverage, frequency, examples, sweep (§6.2.5, §6.3, §6.4) | `python scripts/benchmark_tokenizer.py --language nepali --model tokenizers/nepali/nepali_bpe_8k.model --val <val sample> --train <train sample>` |
| URL discovery counts (§3.3, §3.4) | `python scripts/discover_sitemap_urls.py --domains …` |
| Crawl yields and rejection breakdown (§3.3, §3.4) | `python scripts/collect_manual_web_sitemap.py --language nepali …` |
| Dedup registry size (§4.3) | `python scripts/export_manual_hashes.py --remote gdrive:24KaratAu/nepali/cleaned/manual` |
| Collection ledger (§7.5) | `python scripts/render_collection_ledger.py` |
| Splits, budget, deduplication, final token counts (§5.3) | `notebooks/finalize_{hindi,nepali}_corpus.ipynb` |

Two caveats on exact reproduction. **Sampled figures vary slightly between runs**
where a subcommand samples shards rather than reading all of them — the sampling
basis is stated wherever a number is extrapolated. And
`probe_sitemap_availability.py` is a *shallow* availability check (at most four
sitemaps, four children deep); a domain reading zero there can still yield six
figures under the exhaustive walk in `discover_sitemap_urls.py`, which is why the
two are separate tools.

---

## 8. System Architecture & Engineering Optimizations

Running multi-gigabyte data streaming and tokenizer training on Google Colab free-tier instances required overcoming severe memory and runtime constraints:

1. **Direct File-by-File Streaming**: Bypassed Hugging Face `datasets` C++ memory caching by downloading raw 340MB Parquet files directly via `urllib.request` and streaming records in 5,000-row micro-batches using `pyarrow.parquet.ParquetFile.iter_batches()`.
2. **64-bit Integer Hash Table**: Replaced full 64-character hex strings with 64-bit integer hashes (`int(sha256[:16], 16)`), reducing in-memory deduplication RAM usage by ~75%.
3. **XML Parser Memory Clearing (`root.clear()`)**: Resolved a critical Python `ElementTree.iterparse` memory leak during Wikipedia dump parsing, keeping RAM usage flat under 300MB across 50,000+ XML pages.
4. **Stateful Drive Checkpointing**: Shards are written directly to Google Drive (`data/nepali/cleaned/...`), allowing instant recovery and resumption in the event of Colab session preemption.

### 8.1 Six silent failures, and what they have in common

Each of the following produced wrong output while every check in place reported
success. They are recorded because the shared lesson is more useful than any one
fix: **a transfer that is not verified from the far side, and an assertion made
about a plan rather than a result, will both fail quietly.**

| # | Failure | Symptom | Fix |
| ---: | :--- | :--- | :--- |
| 1 | Shard staging used one bulk `cp … 2>/dev/null` for 220 files | The FUSE transport dropped, the error went to the null device, **2 of 220** automated shards arrived; the corpus came out 95.7% manual with every check green | Copy per file, compare each size against Drive, remount on `Errno 107`, retry |
| 2 | The budget assertion checked that the target was *reachable*, not what was ingested | Failure 1 passed validation and reached Drive | Assert on `written_auto_chars` after Pass 2, before any sync |
| 3 | `rclone cat` piped into a reader | A truncated stream is indistinguishable from a short file: the audit under-counted the corpus by **10,000 documents** and the dedup registry lost 10,000 hashes | Compare bytes read against the size Drive reports; refuse to report a partial count |
| 4 | Split sync used a bulk `cp` with no size check | A failed 3.9 GB `train.txt` write left the *previous run's* file on Drive while `val`/`test` updated normally | Copy and size-verify each split individually |
| 5 | Colab's Drive FUSE accepted a 4.1 GB `train.txt` write and returned the correct size from `os.path.getsize` | The file never committed; `flush_and_unmount()` did not rescue it | Upload through the **Drive API with resumable upload** |
| 6 | The "verified" sync added to fix #5 compared `os.path.getsize(dst)` against the source | It read **the same FUSE cache that dropped the write**. It printed OK for all six Nepali splits while `val.jsonl` and `test.jsonl` never left the mount | Verify against `resp['size']` returned by the Drive API |

Failure 6 is the one worth dwelling on, because it is a fix that recreated the
bug it was fixing. Adding a size check felt like diligence, but the check read
the same layer that had already proved unreliable, so it could only ever confirm
what that layer believed. **Verification has to come from the far side of the
transfer — never from the cache that performed it.** Every check in this
pipeline now reads back from Drive (via `rclone` or the Drive API), not from the
local mount.

A seventh, of a different kind: the Nepali corpus was budgeted using
characters-per-token measured on the **old 32k tokenizer** (5.022) and then
tokenised with the **newly trained 8k** one (4.127). The assertions all passed —
they were checking characters — but the same 2.61B characters produced 632.7M
tokens instead of the projected 520M. The lesson is narrower but real: when a
pipeline both *uses* and *produces* a tokenizer, the budget must be recomputed
against the tokenizer that will actually encode the corpus.

Deduplication had the same shape of flaw (§4.3): it ran manual→automated but
never manual→manual, because each collection run deduplicated against a registry
snapshotted before the following runs existed.

---

## 9. Conclusion & Readiness for Phase 2

Phase 1 deliverables are complete for both languages:

| Deliverable (§1.3 of the project brief) | Hindi | Nepali |
| :--- | :--- | :--- |
| Dataset collection scripts | `scripts/discover_sitemap_urls.py`, `collect_manual_web_sitemap.py --language hindi`, `process_hindi_wikimedia_dump.py` | `scripts/discover_sitemap_urls.py`, `collect_manual_web_sitemap.py --language nepali`, `src/data/collect/` |
| Dataset preprocessing pipeline | `src/data/clean/clean_hindi_web.py`, `hindi_lang_id.py` | `clean_nepali_web.py`, `clean_nepali.py`, `nepali_lang_id.py` |
| Dataset statistics report | §5.2 | §5.1 |
| Train / validation / test splits | `hindi/splits/` on Drive (§7.2) | `nepali/splits/` on Drive (§7.2) |
| Tokenizer training code | `notebooks/finalize_hindi_corpus.ipynb` | `notebooks/finalize_nepali_corpus.ipynb` |
| Vocabulary file | `tokenizers/hindi/hindi_bpe_8k.vocab` (in repo) | `tokenizers/nepali/nepali_bpe_8k.vocab` (in repo) |
| Tokenizer model file | `tokenizers/hindi/hindi_bpe_8k.model` (in repo) | `tokenizers/nepali/nepali_bpe_8k.model` (in repo) |

* Cleaned, deduplicated, leak-free corpora: **520.4M tokens (Hindi)** and 488.5M (Nepali), both above the ~500M target.
* Manual collection at **26.87%** (Hindi) and ~50% (Nepali), both above the 20% floor.
* Zero-unknown-token BPE tokenizers with measured fertility of **1.4938** (Hindi, 8k vocab) and 1.55 (Nepali, 32k vocab).

**Open items carried into Phase 2.**

1. **Numeral convention for the Phase 3 datasets.** `split_digits=True` covers
   ASCII only (§6.1). Measured, the two corpora use opposite conventions — Hindi
   94.8% ASCII, Nepali 99.9% Devanagari — and each is well served as-is, since
   every Nepali value ०–१०० is a single token. Write each reasoning dataset in its
   own corpus's convention and keep quantities to three digits. No retraining
   needed; the failure mode to avoid is writing the Nepali set in ASCII digits,
   which occur 152 times in 11.7M characters of Nepali text.
2. **BPE over Unigram is the weakest-evidenced decision** (§6.2.2). At equal
   vocabulary on equal data, Unigram measured better on both fertility and broken
   graphemes in both languages. BPE was chosen on comparability and convention.
   The ablation that would settle it is Unigram trained on the full split,
   compared on validation loss.
3. **Equalise tokens-seen between the two models.** The Nepali corpus is
   632.7M tokens against Hindi's 520.4M — 1.22x — because its automated budget was
   derived from the old 32k chars/token figure (§5.3). The available *pools* run
   the other way (Hindi 825.8M, Nepali 728.1M), so Hindi remains the
   higher-resource language; the active corpora simply differ in size. Since the
   project's purpose is to compare Model H against Model L, Phase 2 should fix
   `steps x batch_size x sequence_length` identically for both so the comparison
   is not confounded by corpus size. No re-split is needed — Hindi then makes
   roughly one pass over its corpus and Nepali roughly 0.82.
4. **Vocabulary size should be confirmed against `d_model`.** §6.2.4 selects 8,000
   because it is the only size affordable across every width still under
   consideration. Once Phase 2 fixes `d_model`, an 8k-vs-16k comparison on
   validation loss would replace that argument with a measurement.
