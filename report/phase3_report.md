# Phase 3 Report — Reasoning Finetuning, Attention Analysis, and Final Comparison

**Language Models and Agents (CL3-410), Monsoon 2026 — Individual Project**
Model H = Hindi (higher-resource), Model L = Nepali (lower-resource).
Two fully independent monolingual LMs: separate corpora, tokenizers,
vocabularies and weights, at every phase.

> **Status of the results in this report.** Sections 1, 3, 4, 8, 9 and the
> dataset and tokenizer analysis in §5.1 are complete and every number in them
> is measured. The finetuning and attention *results* — §5.2 onward, §6, and
> the parts of §7 that depend on them — are marked **[pending run]** and are
> filled by executing the three Phase 3 notebooks. Each pending block names the
> command that produces it and states, in advance, what each possible outcome
> would mean; nothing there is estimated or assumed.

---

## 1. Executive Summary

Phase 3 finetunes each pretrained model on a synthetic comparative-reasoning
corpus in its own language, measures exact-match accuracy against the
pretrained model, and compares attention before and after.

**What was built.** A generator (`src/reasoning/`) that produces 55,000
examples per language across ten template families, spanning direct comparison
through two-hop transitive chains, with three independent leakage controls. A
supervised finetuning path (`src/train/reasoning_dataset.py`,
`src/train/finetune.py`) that masks the loss to the answer span. An evaluator
(`src/eval/reasoning_eval.py`) that scores exact match under three setups on
three separate test sets, and a pretrain-versus-finetune attention comparator
(`src/eval/attention_reasoning.py`) that reuses the Phase 2 toolkit on
reasoning prompts.

**Three things are measured that a single accuracy number would hide.** The
three test sets of §3.3 — seen entities and wording, unseen entities, unseen
wording — separate "training did not fit" from "it memorised the names" from
"it memorised the template", and they degrade independently. A second seed on
the headline configurations bounds reseeding noise (§8.3). And §5.5 measures
pretraining validation perplexity before and against after, because supervised
finetuning on a narrow task can destroy general next-token prediction — the
course material's worked example moves perplexity 1.33 → 194.7 — and a
reasoning gain reported without that number is an incomplete result.

**Two findings that do not require the finetuning runs**, because they are
properties of the Phase 1 tokenizers measured against the Phase 3 corpus, and
they point in opposite directions:

| | Model H (Hindi) | Model L (Nepali) |
| :--- | ---: | ---: |
| Devanagari-numeral pieces in the 8k vocabulary | 42 | **158** |
| Tokens per stated quantity | 2.044 | **1.458** |
| Corpus fertility (Phase 1, chars/token) | 3.481 | **4.126** |
| Fertility on this synthetic text (chars/token) | 3.111 | 3.252 |
| **Fertility retained out of domain** | **89.4%** | 78.8% |

Nepali segments quantities 29% more efficiently than Hindi — its corpus is
numeral-dense, so BPE learned nearly four times as many multi-digit merges. But
Nepali's tokenizer is also the more corpus-specialised of the two: it loses
21.2% of its fertility on this out-of-domain templated text against Hindi's
10.6%. The lower-resource model therefore enters Phase 3 with an advantage on
the arithmetic-shaped families and a disadvantage on everything else. §7.3
uses the measured per-family accuracies to decide which effect dominates.

**Total project cost so far:** 12.5 GPU-hours of pretraining (Phase 2 §3.6),
plus roughly 0.5 GPU-hours of finetuning and evaluation in this phase.

---

## 2. Where Phases 1 and 2 Left Off

Carried forward because Phase 3's comparisons rest on them. Full detail in
[`report/phase1_report.md`](phase1_report.md) and
[`report/phase2_report.md`](phase2_report.md).

| | Model H (Hindi) | Model L (Nepali) |
| :--- | ---: | ---: |
| Active corpus | 520,409,408 tokens | 632,711,308 tokens |
| Manually collected share | 26.87% | 21.20% |
| Tokenizer | `hindi_bpe_8k` | `nepali_bpe_8k` |
| Fertility (tokens/word) | 1.4938 | 1.5566 |
| Characters per token | 3.481 | 4.126 |
| Trainable parameters | 25,379,840 | 25,379,840 |
| Pretraining tokens (1 epoch) | 468,638,315 | 571,113,472 |
| Test cross-entropy | **3.2885** | 3.4228 |
| Test perplexity | **26.80** | 30.65 |
| Test bits-per-character | 1.3636 | **1.1964** |

The two models are **identical in shape** — same depth, width, heads, context
and parameter count. That was deliberate from Phase 2: it makes every H-versus-L
comparison a statement about the data rather than about capacity.

The Phase 2 conclusion that matters most here is that *which model looks better
depends on the metric*: Hindi wins per-token perplexity by 14.4%, Nepali wins
bits-per-character by 14.0%, and the difference is tokenizer fertility rather
than model quality. Exact-match accuracy on a reasoning task is the first metric
in this project that is **tokenizer-independent** — a correct answer is a
correct answer whatever it cost to encode. §7.2 returns to this.

---

## 3. The Synthetic Reasoning Dataset

### 3.1 Why generated rather than downloaded

The brief requires a synthetic set and forbids an existing benchmark as the
primary finetune data. Beyond compliance there is a practical reason: with
programmatic generation the ground-truth label is known *by construction*, so
accuracy measures the model rather than the annotation. It also makes the
leakage controls in §3.3 enforceable rather than hoped for.

The TA guidance in `doubts-answers.md` settles the distribution question: the
generation script is committed, and the dataset itself goes to Google Drive.
A readable excerpt from the test split is committed at
[`data/hindi/reasoning/sample.md`](../data/hindi/reasoning/sample.md) and
[`data/nepali/reasoning/sample.md`](../data/nepali/reasoning/sample.md), so a
grader can see the data without opening Drive.

### 3.2 Ten template families

Two axes vary independently: what the premises state (explicit quantities, or
inequalities between entities), and what the question asks (which entity, which
extreme, what relation, what rank).

| Family | Entities | Premises | Question | Transitive hops |
| :--- | ---: | :--- | :--- | ---: |
| `rel2_which` | 2 | inequality | which one | 0 |
| `val2_which` | 2 | quantities | which one | 0 |
| `val2_relation` | 2 | quantities | more / less / equal | 0 |
| `rel3_extreme` | 3 | chain | largest / smallest | 1 |
| `rel3_pairwise` | 3 | chain | relation between endpoints | 1 |
| `val3_extreme` | 3 | quantities | largest / smallest | 0 |
| `val4_rank` | 4 | quantities | k-th in descending order | 0 |
| `eq3_relation` | 3 | chain with equality | relation between endpoints | 1 |
| `rel4_extreme` | 4 | chain | largest / smallest | 2 |
| `rel4_pairwise` | 4 | chain | relation between endpoints | 2 |

**The chain families are the point of the exercise.** In `rel3_pairwise` and
`rel4_pairwise` the two entities the question asks about are *never compared to
each other* in the premises. The answer exists only as a consequence of
chaining, so a model that has learned to pattern-match a stated comparison
scores at chance on them while scoring well on `rel2_which`. Reporting accuracy
against hop count separates those two abilities; a single pooled number does
not.

**Surface realisations, and where the coverage is uneven.** Each premise and
question type has several phrasings, and one of each is reserved for the test
split. The relative-premise family carries four — `plain`, `inverted`,
`adjectival` trained, `alt_lexical` held back — which is the full
train-three-hold-one design. The value-premise and superlative-question types
carry only two each, and the `which`, `relation` and `rank` question types carry
**one, with no held-out variant at all**. Every held-out test example therefore
has unseen *premise* wording, but three of the four question types are never
reworded. The unseen-wording number in §5 should be read as a test of premise
phrasing rather than of phrasing in general, and closing the gap is the clearest
single improvement available to this dataset.

Three further details prevent shortcuts:

- **Premise order is shuffled** before rendering. Without it the first premise
  would always mention the largest entity and the task would be solvable from
  position alone.
- **Distinct quantities are separated by a minimum gap** per attribute (3 cm of
  height, 25 rupees of price), so no comparison turns on a one-unit difference.
- **Equality is reachable as an answer, not merely assertable as a premise.**
  With three values and one duplicate the two rank-extremes are never the tied
  pair, so a single tie leaves `बराबर` unreachable; the generator therefore
  collapses the whole chain in a quarter of `eq3_relation` draws, which turns
  the item into transitivity of *equality*. `बराबर` is 12% of relation answers.

### 3.3 Leakage control

Three mechanisms, applied in order, all verified by computation rather than
asserted in prose. The generator exits non-zero if any of them fails.

1. **Held-out entities.** Test examples draw names only from a disjoint pool.
   Hindi and Nepali each have 40 training people and 20 test people, 16 training
   objects and 8 test objects, with **zero overlap**. A model that memorised
   "राम wins" gains nothing: राम never appears in the test split.
2. **Held-out phrasing.** One surface realisation of each premise type and each
   question type is reserved for test and never emitted into train or val.
   1,214 of 5,000 test examples (24.3%) use one. Every example carries the flag,
   so §7 reports accuracy separately for seen and unseen phrasing — two
   different generalisation demands that a pooled number would hide.
3. **Exact-prompt deduplication.** One hash set spans all three splits, seeded
   by generating test first, so a collision is always resolved in favour of
   keeping the test item.

Measured on the shipped corpora, both languages:

| Check | Hindi | Nepali |
| :--- | ---: | ---: |
| Entities shared between train and test | **0** | **0** |
| Prompts shared between train and test | **0** | **0** |
| Prompts shared between val and test | **0** | **0** |
| Prompts shared between train and val | **0** | **0** |

Validation deliberately shares the *train* entity pool. Validation selects
checkpoints, and selecting on out-of-distribution entities would make early
stopping measure generalisation instead of convergence.

**Three test sets, because one number conflates three failures.** The controls
above define a grid, and the corpus ships all three cells of it:

| Split | Entities | Wording | A low score means | Size |
| :--- | :--- | :--- | :--- | ---: |
| `test_seen` | seen | seen | training did not fit the distribution at all | 5,000 |
| `test`, seen phrasing | unseen | seen | it memorised entity names | 3,786 |
| `test`, held-out phrasing | unseen | unseen | it memorised the template | 1,214 |

These degrade independently, so a pooled accuracy does not merely lose
information — it makes three different diagnoses look identical. `test_seen` is
the control: distributionally identical to training, drawn from the train entity
pool with only the shared realisations. Two properties make it trustworthy.
It is deduplicated against train, val and test like everything else, so it is
not a slice of the training data. And unlike `val` — which is also seen/seen,
and was the obvious candidate — it never selected a checkpoint, so its accuracy
carries no selection optimism.

Adding it did not disturb anything: `test_seen` is generated *last*, after
train, so the original three splits draw the identical random stream they drew
before it existed. Their token files are byte-for-byte unchanged, which was
verified by hashing them across the change.

**These are not the only checks.** `tests/test_reasoning_data.py` re-derives
every answer by parsing the rendered Devanagari back into an ordering, rather
than by reading the generator's own metadata. Checking a label against the
metadata only proves the generator agrees with itself; a premise renderer that
stated an inequality backwards would pass that and silently invert part of the
corpus. Parsing the text is the only check that closes it. 26 tests pass.

### 3.4 Corpus statistics

55,000 examples per language, generated with seed 42 and reproducible
byte-for-byte.

| | Hindi train | Hindi val | Hindi test | Nepali train | Nepali val | Nepali test |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Examples | 40,000 | 5,000 | 5,000 | 40,000 | 5,000 | 5,000 |
| Families (each exactly 1/10) | 10 | 10 | 10 | 10 | 10 | 10 |
| Attributes | 6 | 6 | 6 | 6 | 6 | 6 |
| 0-hop / 1-hop / 2-hop | 20k/12k/8k | 2.5k/1.5k/1k | 2.5k/1.5k/1k | 20k/12k/8k | 2.5k/1.5k/1k | 2.5k/1.5k/1k |
| Held-out phrasing | 0 | 0 | 1,214 | 0 | 0 | 1,214 |
| Distinct answer strings | 59 | 59 | 31 | 59 | 59 | 31 |
| Mean prompt (characters) | 114.4 | 114.3 | 116.9 | 114.6 | 114.5 | 115.6 |
| Mean prompt (tokens) | 36.77 | — | 37.91 | 35.24 | — | 35.93 |

Plus `test_seen`: 5,000 examples per language, all ten families, no held-out
phrasing, and 59 distinct answers — the train-pool count, as its purpose
requires.

The distinct-answer counts are a useful consistency check: 40 people + 16
objects + 3 relation words = 59 in train, `val` and `test_seen`, and 20 + 8 + 3
= 31 in `test`, exactly as the disjoint pools require. A `test_seen` that had
leaked held-out names would show 31 or some number between, not 59.

Entity domains carry six attributes each — height, age, weight and marks for
people; price, length and weight for objects — with per-attribute genitive and
copula agreement. Hindi is the harder of the two here: the genitive agrees with
the *possessed* noun, so it is `राम की ऊँचाई` but `राम का वज़न`, and `अंक` is
plural and takes `हैं`. Nepali attaches `-को` throughout, with `-का ... छन्`
for plural `अंक`. Getting this wrong would produce fluent-looking but
ungrammatical text, which is why the agreement table is data in
`entities.py` rather than string formatting spread through the templates.

### 3.5 What the tokenizers do to this corpus

Measured by encoding the generated text with each language's own Phase 1
tokenizer — the same models used in pretraining, unchanged.

| | Model H (Hindi) | Model L (Nepali) |
| :--- | ---: | ---: |
| Mean prompt tokens (train) | 36.77 | 35.24 |
| Longest prompt (tokens) | 67 | 64 |
| Direct answer target (tokens) | 3.37 | 3.51 |
| Rationale answer target (tokens) | 18.65 | 20.17 |
| **Tokens per stated quantity** | 2.044 | **1.458** |
| Devanagari-numeral pieces in vocabulary | 42 | **158** |
| Corpus fertility (chars/token) | 3.481 | **4.126** |
| Fertility on this text (chars/token) | 3.111 | 3.252 |
| **Retained out of domain** | **89.4%** | 78.8% |

**Numerals were rendered in Devanagari, not ASCII**, because the brief requires
natural target-language text. That choice has a measurable consequence and it
is not the one intuition suggests. Neither tokenizer merged multi-digit ASCII
runs — both have exactly ten single-digit ASCII pieces — so ASCII would have
been *uniform* but expensive. Devanagari is neither: Nepali learned 158 numeral
pieces to Hindi's 42, because Nepali web text is dense with Bikram Sambat years
and dates, so a three-digit quantity costs Nepali 1.46 tokens and Hindi 2.04.

The second row of that comparison points the other way. Both tokenizers
fragment this synthetic text more than they fragment their own corpora, which
is expected — templated sentences are out of domain for both. But Nepali loses
**twice as much**: 21.2% of its fertility against Hindi's 10.6%. A tokenizer
trained on a more homogeneous corpus is more specialised to it, and Phase 2
already measured that homogeneity (Nepali's `sangraha` share is 77.5% of
validation documents against Hindi's 71.6%).

So the lower-resource model enters Phase 3 with **a specific advantage on the
four value-premise families**, which are the ones that state quantities, and a
**general disadvantage everywhere else**. This is a prediction with a clean
test: §7.3 reads the per-family accuracies and reports which effect wins,
including the case where neither does.

---

## 4. Finetuning Protocol

### 4.1 What changes from pretraining, and why

Structurally the finetuning loop is the pretraining loop. Three differences are
deliberate, and each changes what is being measured.

**Initialisation, not resumption.** `init_from` loads the Phase 2 checkpoint's
weights only — no optimizer moments, no scheduler state, no step counter. The
pretraining run ended at a 3e-5 learning rate with AdamW second moments shaped
by 14,329 steps of that trajectory; restoring them would damp exactly the
updates this phase depends on. Each model starts from **its own language's**
checkpoint. Nothing crosses between H and L, and the tokenizer and vocabulary
are the Phase 1 ones, unchanged — no tokens added, no vocabulary retrained.

**Loss on the answer span only.** Half of every sequence is the question, which
the model is *given* rather than asked to produce. Training it to predict the
premises teaches it to write comparison questions, not to answer them. Every
prompt position is therefore written to `IGNORE_INDEX`:

```
<bos>  प्रश्न: ... premises ... question ?  \n   उत्तर: राम   <eos>
|------------------ prompt ------------------|--- target ---|
y[i] = IGNORE_INDEX for i < len(prompt)
```

A consequence worth stating plainly: validation loss here is a mean over ~4
scored tokens per example, not over 512, so **it is not comparable to the
pretraining loss curve** and is never plotted against it.

**Selection on exact match, not on loss.** Loss and accuracy come apart on this
task — a model can grow confident about the answer *format* while staying wrong
about the comparison. Exact match is what the brief asks to report, so it is
also what picks the checkpoint. Validation loss is still logged, and the gap
between the two curves is itself reported in §5.2.

### 4.2 Hyperparameters

Identical for both models and both answer styles; the only differences between
the four runs are the language and the target span.

| | Value | Why |
| :--- | :--- | :--- |
| Optimizer | AdamW, β = (0.9, 0.95), ε = 1e-8 | as pretraining |
| Weight decay | 0.1, on matmul weights only | as pretraining |
| Peak learning rate | **1.0e-4** | a third of pretraining's 3e-4; the model is already converged on the language, and 3e-4 overwrites general competence with template-matching inside one epoch |
| Schedule | 100 warmup steps, then cosine to 1e-5 | warmup covers the jump off a cold optimizer |
| Gradient clipping | 1.0 | as pretraining |
| Batch | 64 examples × 1 accumulation | ~3,800 tokens/step |
| Padding | to the longest member of the batch | examples average 57 tokens against a 512 context; padding to `ctx_len` would cost ~7× the compute for nothing |
| Epochs | 3 (1,875 steps) | 625 steps/epoch over 40,000 examples |
| Precision | fp16 + GradScaler | T4 has no bf16 |
| Validation | every 125 steps; loss on all 5,000, exact match on a fixed prefix of 500 | a *fixed* prefix, so the selection metric does not move for reasons unrelated to the model |

**Padding correctness.** Padding sits on the right. Under a causal mask no real
token can attend to a position after it, so the pad ids are unreachable; their
targets are `IGNORE_INDEX`, so they contribute nothing to the loss either. No
attention-mask change was needed. The models use *learned absolute* positions,
which is also why left-padding — the usual batched-generation trick — is not
used anywhere in this phase: it would shift every position index and silently
change the model's input.

### 4.3 Two answer styles

Both were built and both are run, because which supervision a 25M-parameter
model can actually use is an empirical question:

- **`direct`** — the target is `उत्तर: <answer>`, about 3.4 tokens.
- **`rationale`** — the target restates the premise chain and then gives the
  same answer, about 19 tokens: `तर्क: राम > श्याम, श्याम > मोहन` then
  `उत्तर: राम`.

Exact match is scored on the **final answer line in both cases**, so the two
are directly comparable; only the supervision differs. Answer extraction takes
the *last* occurrence of `उत्तर:`, so under the rationale style the model is
scored on its conclusion rather than on anything it restated on the way there,
and under few-shot prompting the worked examples in the preamble cannot be
mistaken for the model's own answer.

### 4.4 Three evaluation setups

| Setup | Checkpoint | Prompt | What it isolates |
| :--- | :--- | :--- | :--- |
| **pretrained, zero-shot** | Phase 2 | bare question | Whether the pretrained model has any notion of the task. Expected near zero; reported anyway, because the *reason* it fails is a formatting failure, which the format rate makes visible. |
| **pretrained, few-shot (k=4)** | Phase 2 | 4 worked examples, then the question | The honest baseline. It supplies the output format, so whatever gap remains is about reasoning rather than about formatting. |
| **finetuned** | Phase 3 | bare question | The format is now learned, so zero-shot is the right comparison. |

Few-shot demonstrations are drawn from the **train** split, so the test split's
held-out entity pool is never shown to the model.

**Format rate is reported alongside accuracy throughout.** A model that answers
wrongly and a model that never emits an answer at all both score 0 exact match,
and they are not the same failure.

Each setup is scored on the three test sets of §3.3 separately, never pooled.

---

### 4.5 How much of the model is updated, and why all of it

Finetuning offers three choices for the trainable set: update everything, freeze
most of it and train a head, or freeze all of it and train small added modules —
the PEFT family, of which LoRA is the best known. **This project updates
everything**, and the reason is a memory calculation rather than a preference.

Full finetuning holds four numbers per parameter: the weight, its gradient, and
AdamW's two moments. At fp32 that is 16 bytes:

| | 25.4M parameters (this project) | 7B parameters |
| :--- | ---: | ---: |
| Weights | 102 MB | 28 GB |
| Gradients | 102 MB | 28 GB |
| AdamW moments (two) | 203 MB | 56 GB |
| **Total before activations** | **406 MB** | **112 GB** |

LoRA exists because the right-hand column does not fit on available hardware,
and because storing one full copy per task is impractical when a copy is tens of
gigabytes. Neither problem exists at 406 MB on a 15 GB T4 — the whole optimizer
state is under 3% of the card. Choosing LoRA here would add a rank
hyperparameter to sweep and a low-rank constraint on the update, in exchange for
saving memory that is not scarce.

There is one genuine argument for PEFT that is not about memory: freezing the
base weights **reduces drift** from the pretrained model, which is the
forgetting measured in §5.5. But it reduces rather than prevents it, and
measuring how much a full finetune actually drifts is more informative here than
constraining it in advance and not knowing. §5.5 reports that measurement.

**What this means for the results.** Every parameter receives gradient from the
answer-span loss, so the reasoning behaviour and whatever happens to general
language modelling are consequences of the same update. They are not separable
by construction, only by measurement.

## 5. Reasoning Results

### 5.1 Finetuning dynamics

**[pending run]** — produced by `notebooks/phase3_colab.ipynb`; curves from
`hindi/logs/finetune_*.jsonl` and `nepali/logs/finetune_*.jsonl`.

The specific thing to read off these curves is **whether validation loss and
validation exact match move together**. They measure different things (§4.1),
and a divergence — loss still falling while exact match has plateaued — would
mean the model is getting more confident about the answer format without
getting more correct, which is the failure mode that motivated selecting on
exact match in the first place.

### 5.2 Pretrained versus finetuned exact match

**[pending run]** — generated verbatim from
`report/figures/reasoning_comparison.md` by
`notebooks/phase3_colab.ipynb`:

```
python -m src.eval.compare_reasoning \
    --inputs hindi/eval/reasoning_*.json nepali/eval/reasoning_*.json \
    --out report/figures/reasoning_comparison.json \
    --md  report/figures/reasoning_comparison.md \
    --figure report/figures/reasoning_accuracy.png
```

Four tables come out of that command. **The three test sets** (§3.3) with a
*memorisation gap* — seen-entity accuracy minus unseen-entity accuracy at the
same wording — and a *template gap* — seen-wording minus held-out-wording at the
same entity pool. Each gap attributes the drop to the one thing that changed
between the columns it spans. **Seed spread**, where a second seed was run
(§8.3). **Pretrained versus finetuned**, with exact match, format rate, and
accuracy split by 0, 1 and 2 transitive hops. And **per-family accuracy** plus
the error analysis of §5.3. Figure: `report/figures/reasoning_accuracy.png`.

**What each shape of result would mean**, stated before the run so the reading
is not fitted to the numbers:

- *Flat across hop counts* — the model learned the relation, and depth is not
  the binding constraint at this scale.
- *A cliff between 1 and 2 hops* — one-step comparison is learned, chaining is
  not. This is the expected outcome for a 25M model and would be the honest
  headline.
- *High at 0 hops only* — the model is pattern-matching a stated comparison
  and does no transitive work at all.
- *Few-shot near finetuned* — finetuning taught format rather than reasoning,
  and the accuracy gain is mostly a formatting gain. The format-rate column
  distinguishes this case directly.

### 5.3 Qualitative successes and failures

**[pending run]** — each `hindi/eval/reasoning_*.json` and
`nepali/eval/reasoning_*.json` carries an `examples` block with 20 correct and
20 incorrect decoded generations, and a `confusions` list of the most frequent
gold → predicted pairs.

Two error signatures are worth separating when these land:

- **Wrong entity, right form** (`राम -> श्याम`) — the model understood what was
  asked and picked the wrong end of the ordering. On `min`-direction questions
  this often means the superlative was ignored.
- **Wrong relation** (`अधिक -> कम`, or anything `-> बराबर`) — the three-way
  relation questions are where a model that has not learned equality will
  concentrate its errors, since `बराबर` is only 12% of relation answers and is
  the easiest class to never predict.

### 5.4 Model H versus Model L on reasoning

**[pending run]** — from the same comparison table.

This is the phase's central comparison, and the one place in the project where
the metric is **tokenizer-independent**: unlike perplexity and bits-per-byte in
Phase 2, a correct answer is a correct answer regardless of how many tokens it
cost to encode. Whatever gap appears here cannot be explained away as a
fertility artefact, which makes it the cleanest resource-tier comparison the
project produces.

The two competing predictions from §3.5 are tested by the *value premises*
versus *relation premises* columns of the generalisation table: Nepali's
numeral advantage predicts it does relatively better on the four value-premise
families, and its weaker out-of-domain fertility predicts the opposite
everywhere else.

---

### 5.5 Did finetuning damage the language model?

Supervised finetuning on a narrow task can destroy general next-token
prediction. This is not hypothetical: the course material's worked example moves
perplexity from **1.33 to 194.7** across an SFT run, a factor of 146. Since §4.5
chose full finetuning precisely so the drift would be measurable rather than
constrained, measuring it is obligatory.

**Method.** The same `src/eval/intrinsic.py` that produced the Phase 2 headline
numbers, pointed at the pretrained and finetuned checkpoints in turn, over the
*pretraining* validation split — ordinary Hindi and Nepali web text, with no
reasoning content in it. Nothing about the metric or the data changes between
the two runs; only the weights do.

A 2M-token slice rather than the full 26M split, via a `--max-tokens` flag added
for this purpose. The flag defaults to off, so the Phase 2 full-split figures
remain reproducible byte-for-byte, and every result records its own budget so a
partial score cannot be mistaken for a complete one. The distinction between
"unchanged" and "destroyed" does not need twelve minutes per checkpoint.

**[pending run]** — `{hindi,nepali}/eval/intrinsic_{pretrained,finetuned}_ft.json`.
Phase 2's full-split baselines for reference: Model H perplexity 26.77 on
validation, Model L 30.64.

What each outcome would mean, stated before the run:

- **Perplexity roughly unchanged** — 1,875 steps at a peak 1e-4 is a small
  perturbation next to 14,329 steps at 3e-4, and the answer-span mask means most
  positions in each sequence contribute no gradient at all. This is the outcome
  the hyperparameters were chosen to produce, and it would say the reasoning
  behaviour was added rather than traded for.
- **Perplexity up by tens of percent** — real drift. The model has moved towards
  the template distribution and away from general text. Worth reporting against
  the reasoning gain as an explicit trade.
- **Perplexity up by orders of magnitude** — the course example's failure mode.
  It would mean the learning rate or epoch count was too aggressive for a model
  this size, and the honest response is to say so rather than to report only the
  reasoning number.

**Why this matters beyond diagnostics.** Phase 2 established that perplexity is
tokenizer-dependent and cannot rank Model H against Model L. That argument is
about *cross-model* comparison. Here the comparison is *within* one model,
before and after, on the same split with the same tokenizer — the one
configuration in which a perplexity difference is unambiguous.

## 6. Attention Analysis, Post-Finetune

### 6.1 Method

`src/eval/attention_reasoning.py` runs both checkpoints over the **same**
reasoning prompts and reports the difference. Two comparability rules are
enforced rather than assumed:

- **Same input.** One prompt list is tokenised once and the resulting batches
  are reused for both checkpoints — the same objects, not merely similar sets.
- **No cross-phase distance comparison.** Mean attention distance scales with
  sequence length, and these prompts are ~37 tokens against Phase 2's 512. The
  distances here are compared pretrained-versus-finetuned only. The JSON records
  the prompt-width distribution so the constraint is visible to a reader who
  might otherwise place the two tables side by side.

Prompts are bucketed by exact token length and batched within a bucket, so
nothing is padded and every position index is the one the model would see
decoding that prompt alone.

**Why reasoning prompts rather than corpus windows.** A comparison over corpus
windows would mostly re-measure Phase 2. The interesting positions are the ones
this task creates and running text does not contain: the second mention of an
entity, where the model must bind it back to its first mention, and the question
tokens, where it must reach back to whichever premise is relevant.

### 6.2 Results

**[pending run]** — `hindi/eval/attention_reasoning.json` and
`nepali/eval/attention_reasoning.json`, with heatmaps at
`report/figures/{hindi,nepali}_reasoning_attention_direct_layer{0,7}.png`.
Layer 0 is the early layer and layer 7 the late one, satisfying the brief's
"at least one early and one late layer per model".

Reported per layer: normalised entropy, mean distance, previous-token mass and
first-position (sink) mass, pretrained against finetuned; plus the twelve heads
whose mean distance moved most, and a confusion table of head roles before
against after using the Phase 2 classifier (`sink`, `previous-token`, `local`,
`diffuse`, `content-based`).

### 6.3 Required discussion — local versus long-range, and head specialisation

**[pending run].** The brief asks specifically whether finetuning changed local
versus long-range attention or head specialisation. The measurements that answer
it, and what each direction would mean:

- **Mean distance increases in late layers** — the model learned to reach back
  to the premises from the question, which is what the task requires. This is
  the outcome consistent with actual chaining.
- **Mean distance decreases, entropy falls** — the model sharpened onto local
  template structure (the fixed `प्रश्न:` / `उत्तर:` frame) rather than onto
  content. Combined with a high format rate and a low exact match, this would
  be direct evidence that finetuning taught format rather than reasoning.
- **Previous-token mass largely unchanged** — the induction-style heads Phase 2
  found are general-purpose and survive finetuning intact; specialisation, if
  any, happened elsewhere.
- **Head roles mostly stable** — 1,875 finetuning steps at 1e-4 is a small
  perturbation next to 14,329 pretraining steps at 3e-4, so *stability is the
  expected result*, and a large reshuffle would be the surprising finding. The
  `roles_changed / heads_total` count is reported so this is a number rather
  than an impression.

---

## 7. The Four Required Questions

### 7.1 How did data scale and quality differ between Model H and Model L?

**Answered in full; this does not depend on the pending runs.**

Not in the direction the labels suggest. "L" denotes Nepali's resource level in
the world, not the size of this corpus: Model L trained on **more** tokens
(571M against 469M) and more characters (2.61B against 1.81B), because Phase 1's
Nepali collection over-delivered.

What differs is **provenance diversity**. Hindi's manually collected share is
26.87% against Nepali's 21.20%, and Nepali is more concentrated in a single
source — `sangraha` accounts for 77.5% of its validation documents against
71.6% for Hindi. A corpus concentrated in one source is more self-similar, and
Phase 2 measured the consequence directly: Nepali achieves the better
bits-per-character (1.1964 against 1.3636) partly because homogeneous text is
easier to model per character, not only because the model is better.

Phase 3 adds a third measurement of the same underlying fact. Nepali's tokenizer
retains only 78.8% of its corpus fertility on out-of-domain synthetic text,
against Hindi's 89.4% — a tokenizer trained on a more homogeneous corpus is more
specialised to it. Corpus homogeneity is therefore visible three separate ways:
in the source mix, in bits-per-character, and in out-of-domain fertility loss.

### 7.2 How do language-modeling and reasoning results compare across the two resource tiers?

**Language modelling: answered. Reasoning: [pending run].**

The Phase 2 language-modelling comparison is genuinely ambiguous, and the
ambiguity is the finding. Hindi wins per-token perplexity by 14.4% (26.80
against 30.65); Nepali wins bits-per-character by 14.0% (1.1964 against 1.3636).
Both are true, and reading either alone gives the wrong answer, because Nepali's
tokenizer packs 18.6% more text into each token — so predicting a Nepali token
is a strictly harder problem, which perplexity charges it for and
bits-per-character credits it for.

**Reasoning accuracy resolves that ambiguity in a way no Phase 2 metric could**,
because exact match is tokenizer-independent. This is the single most valuable
comparison in the project, and the pending run supplies it. Whatever the
direction, the result is interpretable: a gap here cannot be attributed to
fertility, so it points at the corpus or the model rather than at the encoding.

### 7.3 What tokenizer / corpus factors most affected the lower-resource model?

**Substantially answered; one sub-question pending.**

Three measured factors, two of which are already settled:

1. **Fertility (settled, Phase 2).** Nepali's 4.126 characters per token against
   Hindi's 3.481 is why the two models' perplexities are not comparable, and why
   the brief's bits-per-byte requirement exists. It penalises Nepali on every
   per-token metric while being a property of the tokenizer, not the model.
2. **Numeral segmentation (settled, Phase 3).** Nepali's vocabulary holds 158
   Devanagari-numeral pieces to Hindi's 42, so a stated quantity costs 1.46
   tokens against 2.04 — a 29% advantage to the lower-resource model, arising
   from Nepali web text being dense with Bikram Sambat dates. This is the one
   place in the project where the lower-resource tokenizer is unambiguously
   better, and it is worth stating because it cuts against the expected story.
3. **Out-of-domain robustness (settled, Phase 3).** Nepali loses 21.2% of its
   fertility on this synthetic text against Hindi's 10.6%. Its tokenizer is
   more corpus-specialised, which is a direct cost of corpus homogeneity.

Factors 2 and 3 point in **opposite directions**, and which dominates is the
pending sub-question: the value-premise versus relation-premise split in the
generalisation table decides it. If Nepali's relative accuracy is better on the
four value-premise families than on the six relation-premise ones, the numeral
advantage is doing real work; if not, out-of-domain fragmentation dominates and
the numeral advantage is too small a share of the sequence to matter. A third
outcome — no difference between the splits — would say the models are not
limited by tokenization here at all, which given 25M parameters and two-hop
chains is a live possibility.

### 7.4 What evidence explains the observed differences?

**Partly answered; completed by the pending runs.**

Evidence that already exists, and its strength:

| Claim | Evidence | Strength |
| :--- | :--- | :--- |
| The two models differ in data, not capacity | Identical architecture, 25,379,840 parameters each, identical hyperparameters | **Controlled by construction** |
| Perplexity favours H for tokenizer reasons | Fertility 3.481 vs 4.126 chars/token, measured on the split files | Strong, directly measured |
| Nepali's corpus is more homogeneous | Manual share 21.20% vs 26.87%; `sangraha` 77.5% vs 71.6% of validation docs | Strong, but correlational |
| Homogeneity makes text easier to model | Nepali's better bits-per-character | **Suggestive only** — corpus size and domain mix differ simultaneously |
| Nepali's tokenizer is more specialised | 21.2% vs 10.6% out-of-domain fertility loss | Strong, directly measured, new in Phase 3 |
| Training-step cost does not differ by language | Throughput 36,898 vs 36,457 tok/s, a 1.2% spread on identical shapes | **Clean negative control** (Phase 2 §3.6) |

**The honest limit.** With two models there are two data points, and every
distinguishing factor — corpus size, provenance mix, fertility, domain balance —
differs simultaneously. Nothing in this project can attribute an H-versus-L gap
to a single cause, and no result here has error bars: each number is one run at
one seed. The per-hop and per-family breakdowns are the closest thing to a
controlled comparison available, because within a single model they vary
reasoning depth and premise form while holding the corpus, tokenizer and weights
fixed.

---

## 8. Reproducibility

Every artefact in this phase carries a provenance block: the exact command, the
git commit and branch, whether the tree was dirty, library versions, and a
SHA-256 for each named input.

### 8.1 Full pipeline

```bash
# 1. Reasoning corpora (~10 s per language, CPU only, no GPU needed).
#    --test-seen adds the seen-entities/seen-wording control split of SS3.3.
python -m src.reasoning.generate --language hindi --out-dir data/hindi/reasoning \
    --train 40000 --val 5000 --test 5000 --test-seen 5000 --seed 42
python -m src.reasoning.generate --language nepali --out-dir data/nepali/reasoning \
    --train 40000 --val 5000 --test 5000 --test-seen 5000 --seed 42

# 2. Verify labels and leakage before spending GPU time
python -m pytest tests/ -q

# 3. Finetune (~6 min per run on a T4)
python -m src.train.finetune --config hindi/configs/finetune.yaml
python -m src.train.finetune --config hindi/configs/finetune_rationale.yaml
python -m src.train.finetune --config nepali/configs/finetune.yaml
python -m src.train.finetune --config nepali/configs/finetune_rationale.yaml

# 4. Evaluate: three setups per model, on the held-out-entity test split
python -m src.eval.reasoning_eval --checkpoint <pretrained> --mode zeroshot ...
python -m src.eval.reasoning_eval --checkpoint <pretrained> --mode fewshot --shots 4 ...
python -m src.eval.reasoning_eval --checkpoint <finetuned> --mode finetuned ...

# 4b. The same finetuned checkpoint on the control split (SS3.3), so the three
#     test-set numbers can be read side by side
python -m src.eval.reasoning_eval --checkpoint <finetuned> --mode finetuned \
    --test data/hindi/reasoning/test_seen.jsonl ...

# 4c. Forgetting: pretraining validation perplexity before and after (SS5.5).
#     --max-tokens defaults to off, so Phase 2's full-split numbers still
#     reproduce exactly.
python -m src.eval.intrinsic --checkpoint <pretrained> --splits val \
    --max-tokens 2000000 --tokens-dir <drive>/hindi/tokens --language hindi ...
python -m src.eval.intrinsic --checkpoint <finetuned> --splits val \
    --max-tokens 2000000 --tokens-dir <drive>/hindi/tokens --language hindi ...

# 5. Attention, pretrained vs finetuned
python -m src.eval.attention_reasoning --pretrained <p> --finetuned <f> ...

# 6. Consolidated tables, error analysis and figure
python -m src.eval.compare_reasoning --inputs */eval/reasoning_*.json ...
```

All of the above runs end-to-end in one notebook on the platform actually
used: [`notebooks/phase3_colab.ipynb`](../notebooks/phase3_colab.ipynb), a
single Colab T4 session of about 55 minutes. Unlike Phase 2, which split the
two models across Colab and Kaggle because Nepali pretraining alone ran 4.87
hours, Phase 3's whole GPU cost is under an hour — so both models are finetuned
in the same session from the same code copy, which is a tighter guarantee of
identical treatment than two platforms could give.

### 8.2 Determinism

The corpora derive entirely from `--seed` through a single `random.Random`;
regenerating with the same seed reproduces them byte-for-byte, and
`test_generation_is_deterministic_under_a_seed` asserts it. Finetuning is seeded
and uses the same resume-capable checkpoint format as pretraining, so an
interrupted Colab session continues from `finetune_latest.pt` rather
than restarting.

### 8.3 Seeds

Every configuration is trained at seed 42. `direct`-style runs are additionally
trained at seed 7, redirected to their own checkpoint directory and log through
`--override`, so the two do not overwrite each other:

```bash
python -m src.train.finetune --config hindi/configs/finetune.yaml \
    --override seed=7 checkpoint.dir=<drive>/hindi/checkpoints_ft_direct_s7 \
      log.jsonl=<drive>/hindi/logs/finetune_direct_s7.jsonl
```

Two seeds is not an error bar. It gives a *range*, which is the minimum needed
to know whether the per-family and per-hop differences in §5 are larger than
reseeding noise — a three-point gap means nothing if reseeding moves the headline
by five. `compare_reasoning` reports the range where a second seed exists and
says so explicitly where it does not.

### 8.4 Test coverage

`python -m pytest tests/ -q` — 28 tests run without torch (the reasoning corpus,
its labels, its leakage controls, the control split's distribution, and answer
extraction), and the remaining
suites covering model shapes, causal masking, checkpoint resume, the packed
dataset and the finetuning mask run wherever torch is installed. The torch
imports in the Phase 2 suites were made conditional in this phase so the whole
suite still collects on a machine without a GPU stack.

---

## 9. Limitations

1. **Two models, and a range rather than error bars.** The `direct`
   configurations are run at two seeds and report a range (§8.3); the
   `rationale` configurations are single runs. Two seeds bound reseeding noise
   but do not give a confidence interval, and every H-versus-L difference
   remains a single pair of observations whose explanatory factors are
   confounded (§7.4).
2. **The reasoning task is narrow.** Comparative and transitive ordering over
   templated sentences is a small, well-defined slice of reasoning. Success
   here is evidence about ordering relations, not about reasoning generally,
   and the templates are the model's entire exposure to the task.
3. **Synthetic text is out of domain for both tokenizers**, measurably so
   (§3.5). Some of the difficulty in this phase is the encoding of templated
   sentences rather than the reasoning in them, and the two effects are not
   separable with the evidence collected here.
4. **Held-out phrasing is a weaker control than held-out entities, and its
   coverage is uneven.** The reserved realisations are lexical variants within
   the same frames, so a model could generalise across them without
   generalising to genuinely new phrasing. Worse, the coverage is not uniform:
   relative premises carry the full four forms with one held back, but the
   `which`, `relation` and `rank` question types carry a single form and are
   never reworded (§3.2). The unseen-wording column therefore tests premise
   phrasing, not question phrasing. Entity disjointness is the stronger
   guarantee, and it is exact.
5. **25M parameters is small for multi-hop reasoning.** A ceiling on the 2-hop
   families would be an unremarkable result at this scale and should not be read
   as a statement about the languages or the corpora.
6. **Attention statistics describe where the model looks, not why.** Head
   labels come from crude thresholds (Phase 2 §6.1) chosen to make 64 heads
   discussable, not to make a scientific claim; a change in mean distance is
   consistent with several mechanisms and identifies none of them.
7. **The forgetting measurement is a 2M-token slice**, not the full 26M
   validation split (§5.5). That is ample to separate "unchanged" from
   "degraded", but a small real drift and sampling noise are not distinguishable
   at that budget, and the figures are not directly comparable to Phase 2's
   full-split numbers.

---

## 10. Conclusion

**[to be completed once §5 and §6 are populated.]** The determinate part:
Phase 3 delivers two independent synthetic reasoning corpora with verified
labels and three enforced leakage controls, a finetuning path that supervises
the answer span alone, an evaluation that separates formatting failure from
reasoning failure, and an attention comparison that holds its input fixed
across checkpoints. It also adds a measured tokenizer finding that Phase 2
could not see: the lower-resource tokenizer is simultaneously *better* at
numerals (1.46 against 2.04 tokens per quantity) and *less robust* off its own
corpus (78.8% against 89.4% fertility retained) — two consequences of the same
corpus homogeneity, pulling in opposite directions.
