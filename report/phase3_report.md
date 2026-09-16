# Phase 3 Report — Reasoning Finetuning, Attention Analysis, and Final Comparison

**Language Models and Agents (CL3-410), Monsoon 2026 — Individual Project**
Model H = Hindi (higher-resource), Model L = Nepali (lower-resource).
Two fully independent monolingual LMs: separate corpora, tokenizers,
vocabularies and weights, at every phase.

> **Status of the results in this report.** Every number is measured; nothing
> is estimated, and no section is outstanding.
>
> Two configurations are reported. §5.2-5.4 and §6 describe the finetune
> specified in §4.2 (3 epochs at 1e-4). §5.6 reports a gentler one (1 epoch at
> 3e-5) that turned out better on every axis, and is what the conclusions treat
> as the result.
>
> Several sections of §5 and §6 stated, before the runs, what each possible
> outcome would mean. Where a prediction was contradicted — §4.1's expectation
> that 1,875 steps would be "a small perturbation", §5.2's pre-registered
> reading of the hop-count curve, §3.3's justification for the validation
> split — the text now says so rather than being quietly revised. Those
> paragraphs are the most useful part of this report.

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

**The headline result.** Finetuning teaches both models the answer *format*
almost perfectly and the *task* only partly. The clearest single number in the
report is that the *pretrained* Model H, given four worked examples, produced a
correctly formatted answer **99.98%** of the time and got **0.00%** of them
right. Format and reasoning are separable, and here they separate completely.

**The second result is that our own first configuration was simply wrong, and
the report says so.** §4.2 specified 3 epochs at a peak 1e-4 and argued for it.
That run reached exact match 0.2458 (H) / 0.2144 (L) while destroying most of
each model's general language ability -- validation perplexity rose 55x and
118x, leaving 33% and 18% of what pretraining had taught them. Repeating it at
**1 epoch and 3e-5** was better on every axis at once: accuracy **0.2992** /
**0.2320**, perplexity within **1.2x** of the pretrained model, **96%** of the
language model retained, in a third of the time. There was no trade-off to
find -- only a wasteful recipe (§5.6).

**Both configurations fail the same way**, and the failure is diagnosable
because of the design rather than the numbers. Both models learned to answer
"who?" with a name from the *training* pool, so questions whose answer is a
held-out name score near zero while three-way relation questions score 0.45-0.67
against a 0.33 baseline. The aggressive recipe could not name an unseen entity
at all; the gentle one sometimes can -- and every bit of its accuracy advantage
comes from exactly those families, which is what ties the forgetting and the
memorisation together into one mechanism (§5.6).

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

### 1.1 What happened, in plain terms

This section deliberately avoids jargon. Everything in it is restated precisely,
with the numbers, in §5 and §6.

**What we tried to do.** Each model had already been trained to predict ordinary
Hindi or Nepali text. We then trained it further on 40,000 made-up comparison
questions in its own language — *"Ram is taller than Shyam, Shyam is taller than
Mohan, who is tallest?"* — and checked whether it could answer questions of the
same kind about **people it had never seen before**.

**What the models learned.** The shape of an answer, almost perfectly. Asked a
question, they reply `उत्तर:` followed by a name, in the right format, 96-100% of
the time. What they mostly did not learn is *which* name.

**The specific mistake they make.** They only ever answer with names from the
training set. Ask Model H about भरत (a name held back for testing) and it answers
सीता (a name it trained on). Sometimes it answers a question about a person with
the name of an object. Because the test set deliberately uses only names the
models never saw, a question whose answer is a name is one they can essentially
never get right — the score on those is 0%.

**Where they do succeed.** On questions whose answer is one of three fixed words
— "more", "less" or "equal" — they score 45-67%, against 33% for guessing. So the
models did learn something real about comparison. It only shows up where the
answer does not require naming an entity.

**The cost, at our first settings.** Teaching them this damaged their ordinary
language ability badly.
Before finetuning, Model H predicted ordinary Hindi text with a perplexity of 20;
afterwards, 1,115 — about 55 times worse. Model L went from 24 to 2,802, about
118 times worse. Both are still slightly better than random guessing, but only
just. This is a known effect with a name, *catastrophic forgetting*, and the
course material warns about it explicitly.

**Then we tried training much more gently** -- one pass over the data instead of
three, at a third the learning speed -- to see what preserving the language model
would cost in accuracy. It cost nothing. The gentler version was **better at
everything**: more questions right (30% vs 25% for Model H), the language ability
almost untouched (96% kept instead of 33%), and three times faster to train. Our
first settings were simply too aggressive, and §5.6 reports both so the
comparison is on the record.

**And the two problems turned out to be one problem.** The aggressively trained
models could never name a person they had not seen before. Naming someone
mentioned in the question means copying a word out of the text you were given --
which is a language ability, and that is what the aggressive training destroyed.
The gently trained models kept their language ability, and with it they can
sometimes copy the right name. All of the gentle version's extra accuracy comes
from exactly those questions.

**Why we can say all this.** Three design decisions made the diagnosis possible,
and without them the result would have looked like a single unexplained number:

- **Separate name lists for training and testing.** This is what exposed the
  memorisation. With shared names the models would have scored far higher and we
  would have learned nothing.
- **Three separate test sets** instead of one score (§3.3). The models get 62%
  (H) and 45% (L) on questions about *familiar* names, against 25% and 21% on
  unfamiliar ones. That gap is the memorisation, measured.
- **Reporting how often the model produced a well-formed answer at all**, not
  just how often it was right. The clearest single number in this report is that
  the *pretrained* model, shown four worked examples, produced a correctly
  formatted answer 99.98% of the time and got 0.00% of them right. Format and
  reasoning are separate things, and that number separates them.

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

### 2.1 The bonus ablation, for completeness

The brief places the optional ablation in a project-level section of its own,
after the three phases, and asks for it "in the same project repository". It is
reported in full in [`report/phase2_report.md`](phase2_report.md) §7, because it
reuses the Phase 2 evaluation suite and compares against the Phase 2 Hindi
model. It is summarised here so that this report consolidates the whole project
as §3.3 of the brief requires.

**Model H retrained with the positional embedding removed**, matched on seed,
steps, batch composition and learning-rate schedule -- the only difference being
`pos_encoding: learned` against `none`:

| Hindi test split | Learned positions | No positions | Change |
| :--- | ---: | ---: | ---: |
| Cross-entropy (nats) | 3.2885 | 3.3216 | +0.0331 |
| **Perplexity** | **26.80** | **27.70** | **+3.4%** |
| Bits-per-character | 1.3636 | 1.3773 | +1.0% |

**Removing positions entirely costs 3.4% perplexity, far less than expected, and
the reason is the finding.** Under a causal mask the query at position *i*
attends over exactly *i + 1* keys, so the count of visible tokens is itself a
weak monotonic position signal that no ablation removes. The measured gap is
"learned absolute positions versus whatever the causal mask leaks", not
"positions versus none". For scale, the distance from the baseline to a model
that has learned nothing (`ln 8000 = 8.99`) is 5.70 nats; the ablation gives up
0.033 of that, or **0.6% of everything the baseline learned**. Generation
metrics and attention statistics are essentially unchanged.

Per the TA guidance in `doubts-answers.md`, the bonus is submitted on its own
branch built on top of `phase-3`, so that branch carries all three phases plus
the ablation.

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

### 5.1 Finetuning dynamics -- and a selection metric that rewarded the wrong thing

Curves from `{hindi,nepali}/logs/finetune_*.jsonl`. Validation loss every 125
steps, exact match on a fixed 500-example prefix at the same interval.

| Run | val loss, step 125 -> 1875 | val EM, step 125 -> 1875 | best val EM | minutes |
| :--- | :--- | :--- | ---: | ---: |
| H, direct | 0.331 -> 0.155 | 0.382 -> 0.626 | 0.6380 | 4.8 |
| H, rationale | 0.234 -> 0.015 | 0.380 -> 0.916 | **0.9240** | 9.8 |
| L, direct | 0.329 -> 0.210 | 0.404 -> 0.514 | 0.5140 | 4.9 |
| L, rationale | 0.216 -> 0.017 | 0.370 -> 0.924 | **0.9240** | 9.6 |

**Nothing in these curves looks wrong.** Validation loss falls monotonically,
validation exact match rises monotonically, and neither turns over by step 1,875
in any of the four runs. There is no overfitting signal. §5.1's stated purpose
was to check whether loss and accuracy diverge; they do not -- they agree
throughout, and they agree on the wrong answer.

**The divergence is between validation and test, and it is enormous:**

| Run | best val EM | test EM (unseen entities) | gap |
| :--- | ---: | ---: | ---: |
| H, direct | 0.6380 | 0.2458 | 0.392 |
| H, rationale | **0.9240** | **0.1794** | **0.745** |
| L, direct | 0.5140 | 0.2144 | 0.300 |
| L, rationale | 0.9240 | 0.1992 | 0.725 |

The `rationale` runs look like triumphs on validation -- 92.4% exact match,
validation loss 0.015 -- and are the *worst* two runs on test.

**Why: the validation split shares the training entity pool, so selecting on it
selected for memorisation.** §3.3 justified that choice on the grounds that
"selecting on out-of-distribution entities would make early stopping measure
generalisation instead of convergence". That reasoning is defensible in the
abstract and was wrong in practice: it made the selection metric blind to the
one failure that dominates these results. A checkpoint chosen on `test_seen`-like
data cannot distinguish a model that learned the relation from one that learned
the name list, because on that data both score well.

**What should have been done.** Select on a validation split drawn from a *third*
entity pool -- disjoint from both train and test. That keeps selection
out-of-distribution with respect to memorisation while keeping the test set
untouched for reporting. The entity inventories have room: 60 people and 24
objects per language, currently split 40/20 and 16/8. This is recorded as
limitation 8 rather than fixed, because changing it now would invalidate every
number above.

**The cheap diagnostic that would have caught it during training.** Log exact
match on a held-out-entity sample alongside the validation number. The gap
between the two is visible from the first evaluation and needs no extra
training; it is three lines in `validate_em`.

### 5.2 Pretrained versus finetuned exact match

Full tables in [`report/figures/reasoning_comparison.md`](figures/reasoning_comparison.md);
figure in [`report/figures/reasoning_accuracy.png`](figures/reasoning_accuracy.png).

| Model | Style | Setup | Exact match | Format rate |
| :--- | :--- | :--- | ---: | ---: |
| H | direct | pretrained, zero-shot | 0.0000 | 0.0040 |
| H | direct | pretrained, few-shot (k=4) | 0.0000 | **0.9998** |
| H | direct | finetuned | **0.2458** | 0.9614 |
| H | direct | finetuned, seed 7 | 0.2584 | 0.9760 |
| H | rationale | finetuned | 0.1794 | 0.8686 |
| L | direct | pretrained, zero-shot | 0.0000 | 0.0000 |
| L | direct | pretrained, few-shot (k=4) | 0.0000 | 0.5374 |
| L | direct | finetuned | **0.2144** | 1.0000 |
| L | direct | finetuned, seed 7 | 0.2170 | 1.0000 |
| L | rationale | finetuned | 0.1992 | 0.9716 |

**The format rate is the most informative column in this report.** Model H's
pretrained checkpoint, shown four worked examples, produced a correctly
formatted answer **99.98%** of the time and scored **0.0000** exact match. Four
demonstrations were enough to teach the output format completely and the task
not at all. §4.4 predicted that few-shot prompting would separate formatting
from reasoning; this is a cleaner separation than expected -- the two are not
merely distinguishable here, they are fully decoupled.

Zero-shot scores 0.0000 at a 0.0040 format rate, so the pretrained model has no
notion of the task *or* its shape, exactly as predicted.

**Accuracy against reasoning depth is not monotonic**, and the reason is not
depth:

| Hops | Model H | Model L |
| ---: | ---: | ---: |
| 0 | 0.1480 | 0.1072 |
| 1 | **0.3933** | **0.3713** |
| 2 | 0.2690 | 0.2470 |

One hop beats zero. §5.3 shows why: `hops` is confounded with *what the answer
is*, and the answer type dominates everything else. The pre-registered readings
in the previous draft of this section -- "a cliff between 1 and 2 hops",
"high at 0 hops only" -- do not describe these data, and the family breakdown
explains why.

**Both answer styles, resolved.** `rationale` loses to `direct` in both
languages (0.1794 vs 0.2458; 0.1992 vs 0.2144) and has a lower format rate. At
25M parameters the longer target -- restate the chain, then answer -- is harder
to produce than the answer alone, and the extra supervision does not pay for
itself. §4.3 built both because this was an empirical question; the answer is
that the simpler target wins.

**Seeds.** Range 0.0126 (H) and 0.0026 (L) across two seeds, so the H-vs-L gap
of 0.0314 is about 2.5x the reseeding noise on Model H's side -- real, but not
comfortably so, and stated that way in §7.2.

### 5.3 Qualitative successes and failures -- the answer type is everything

Per-family accuracy, finetuned `direct`, sorted:

| Family | Answer is | Hops | Model H | Model L |
| :--- | :--- | ---: | ---: | ---: |
| `eq3_relation` | relation word | 1 | 0.6660 | 0.6600 |
| `rel4_pairwise` | relation word | 2 | 0.5280 | 0.4940 |
| `val2_relation` | relation word | 0 | 0.4760 | 0.5360 |
| `rel3_pairwise` | relation word | 1 | 0.4640 | 0.4540 |
| `rel2_which` | **entity name** | 0 | 0.2640 | 0.0000 |
| `rel3_extreme` | **entity name** | 1 | 0.0500 | 0.0000 |
| `rel4_extreme` | **entity name** | 2 | 0.0100 | 0.0000 |
| `val2_which` | **entity name** | 0 | 0.0000 | 0.0000 |
| `val3_extreme` | **entity name** | 0 | 0.0000 | 0.0000 |
| `val4_rank` | **entity name** | 0 | 0.0000 | 0.0000 |

The table splits perfectly on one variable, and it is not reasoning depth or
premise form. **Where the answer is one of three relation words, both models
score 0.45-0.67 against a 0.33 chance baseline. Where the answer is an entity
name, they score at or near zero.**

**The mechanism, read off the decoded generations.** Every wrong answer on an
entity-naming family is a *training-pool* name:

| Gold (held-out pool) | Predicted (train pool) | Raw output |
| :--- | :--- | :--- |
| भरत | सीता | `उत्तर: सीता` |
| अरुण | मोहन | `उत्तर: मोहन` |
| करण | आनंद | `उत्तर: आनंद` |
| तनुजा | रस्सी | `उत्तर: रस्सी` |
| ताला | अंजलि | `उत्तर: अंजलि` |

The outputs are well formed and the answer marker is always present, so this is
not an extraction failure. The model learned *"when asked who, emit one of the
56 entity names I saw during training"*. The last two rows show it does not even
respect the entity domain: तनुजा is a person and रस्सी is a rope; ताला is a
padlock and अंजलि is a person.

Because the test split draws only from the disjoint held-out pool, a
name-answering question is one the model can essentially never get right. The
exact 0.0000 in three families is therefore a real property of the model, not a
bug -- and the control split confirms it from the other side: on `test_seen`,
where the names *are* from the training pool, Model H scores 0.6226.

**This is the failure the course material describes**, reproduced at 31x the
parameter count of its demonstration model (25.4M against 824,064): "the model
learned to answer with one of the twenty names it had trained on", training loss
low, exact match on unseen names zero. Our numbers are not as extreme -- the
relation-word families genuinely work -- but the mechanism is identical.

**Relation errors are directional, not formatting.** The most frequent single
confusion for Model H is `अधिक -> कम` (559 times) and its inverse `कम -> अधिक`
(276). The model produces a valid relation word and picks the wrong direction.
Confusions into `बराबर` are rarer (45 + 43), consistent with equality being 12%
of relation answers and the easiest class to under-predict.

### 5.4 Model H versus Model L on reasoning

This is the phase's central comparison, and the only place in the project where
the metric is **tokenizer-independent**: unlike perplexity and bits-per-byte in
Phase 2, a correct answer is correct regardless of how many tokens it cost to
encode. A gap here cannot be explained away as fertility.

| | Model H (Hindi) | Model L (Nepali) |
| :--- | ---: | ---: |
| Exact match, unseen entities, all wording | **0.2458** | 0.2144 |
| Exact match, unseen entities, seen wording | 0.2456 | 0.2116 |
| Exact match, seen entities (`test_seen`) | **0.6226** | 0.4542 |
| **Memorisation gap** (matched wording) | **0.3770** | **0.2426** |
| Template gap (unseen wording) | -0.0007 | -0.0116 |
| Seed range | 0.0126 | 0.0026 |
| Forgetting (val perplexity) | 55x | 118x |

The memorisation gap is taken between the two *seen-wording* columns, so the
only thing differing across it is the entity pool. Against the pooled
unseen-entity figure it would be 0.3768 -- the distinction is immaterial here
but the matched comparison is the defensible one.

**Model H is ahead on every axis, and by a consistent margin.** It is 14.7%
better on unseen entities, 37% better on seen ones, and was damaged half as much
as Model L. Since the two models are identical in architecture and
hyperparameters, this is a statement about the corpora.

**The memorisation gap is the more interesting comparison.** Model H memorised
*more* (0.3770 against 0.2426) and still generalised better. Those are not in
tension: Model H learned the training names more thoroughly, and it also learned
more of the relation. Model L learned less of both.

**The template gap is ~0 in both, and slightly negative.** Held-out premise
phrasing costs nothing measurable -- Model H scores 0.2456 on seen wording and
0.2463 on unseen. The models generalise across phrasing and fail across
entities, which is the opposite of what a template-matching account would
predict for phrasing and exactly what it predicts for entities. Note the caveat
from §3.2: three of the four question types are never reworded, so this column
tests premise phrasing only.

**What this does not settle.** Two models, two seeds each, and every
distinguishing factor -- corpus size, provenance mix, fertility, domain balance
-- differs at once. The Model H advantage on unseen entities (0.0314) is about
2.5x its own seed range (0.0126), which supports the direction but not a precise
magnitude.

### 5.5 Did finetuning damage the language model? -- yes, severely

**Method.** `src/eval/intrinsic.py`, the Phase 2 evaluator, pointed at the
pretrained and finetuned checkpoints in turn over each model's *pretraining*
validation split -- ordinary web text with no reasoning content. Only the
weights differ between the two runs. A 2M-token slice via `--max-tokens`, so the
figures are not interchangeable with Phase 2's full-split numbers; the
before/after comparison is valid because both sides use the same slice.

| | Model H | Model L |
| :--- | ---: | ---: |
| Cross-entropy before (nats) | 3.0053 | 3.1682 |
| Cross-entropy after (nats) | 7.0168 | 7.9380 |
| **Perplexity before** | **20.19** | **23.77** |
| **Perplexity after** | **1,115.21** | **2,801.61** |
| Degradation | **55x** | **118x** |

Uniform guessing over an 8,000-token vocabulary is 8.987 nats. Measured against
that floor:

| | learned over uniform, before | after | **retained** |
| :--- | ---: | ---: | ---: |
| Model H | 5.982 nats | 1.970 | **33%** |
| Model L | 5.819 nats | 1.049 | **18%** |

**Model H threw away two-thirds of everything pretraining taught it, and Model L
four-fifths.** Both remain better than random, but barely. For comparison, the
course material's worked example degrades 146x; ours are 55x and 118x, the same
phenomenon at the same order of magnitude.

**Why it is this severe.** Four choices compound, all of them made in
`finetune.yaml`:

1. **Full finetuning** -- every one of the 25.4M parameters moved. §4.5 argued
   for this on memory grounds and stated the trade explicitly: freezing weights
   "reduces rather than prevents" drift, and "measuring how much a full finetune
   actually drifts is more informative here than constraining it in advance and
   not knowing". This is that measurement, and it does not vindicate the choice.
2. **1,875 steps at a peak 1e-4.** §4.1 called this "a small perturbation next
   to 14,329 steps at 3e-4". It is 13% as many steps at a third the peak rate --
   not small enough, and that prediction was wrong.
3. **A very narrow target distribution** -- 40,000 examples from ten templates,
   seen three times each.
4. **No pretraining data mixed back in**, which is the standard mitigation.

The answer-span mask narrows the gradient *signal* but not its *reach*: every
parameter still receives it, exactly as §4.5 said.

**This is the clean case for a perplexity comparison.** Phase 2 established that
perplexity is tokenizer-dependent and cannot rank Model H against Model L. That
argument is about *cross-model* comparison. Here the comparison is within one
model, before and after, on the same split with the same tokenizer -- the one
configuration in which a perplexity difference is unambiguous.

### 5.6 A gentler recipe -- there was no trade-off, only a worse recipe

§5.5 showed the configuration in §4.2 made a bad trade. The obvious question was
whether a better trade existed, so the finetune was repeated at **1 epoch and a
peak 3e-5** -- a third the learning rate, a third the steps, `direct` style
only, everything else fixed through `--override`.

The expectation was a trade-off curve. There is no curve, because the gentle
recipe is **better on both axes at once**:

| Model | Recipe | Reasoning EM | Format rate | val PPL | vs pretrained | LM retained |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| H | pretrained, no finetune | 0.0000 | 0.004 | 20.19 | 1.0x | 100% |
| H | 3 epochs @ 1e-4 | 0.2458 | 0.961 | 1,115.21 | 55.2x | 33% |
| H | **1 epoch @ 3e-5** | **0.2992** | **0.994** | **25.00** | **1.2x** | **96%** |
| L | pretrained, no finetune | 0.0000 | 0.000 | 23.77 | 1.0x | 100% |
| L | 3 epochs @ 1e-4 | 0.2144 | 1.000 | 2,801.61 | 117.9x | 18% |
| L | **1 epoch @ 3e-5** | **0.2320** | **1.000** | **29.40** | **1.2x** | **96%** |

The gentle recipe is **more accurate** (+0.053 Hindi, +0.018 Nepali), has an
**equal or better format rate**, retains **96%** of pretraining against 33% and
18%, and trains in **1.5 minutes instead of 4.8**. Of the three outcomes §5.6
registered in advance, this is the first -- "the aggressive recipe was simply
wasteful" -- and more strongly than that prediction allowed, since accuracy
went *up* rather than staying similar.

**The aggressive configuration was straightforwardly overfitted.** §5.1 found no
overfitting signal in the validation curves, and that is still true -- but the
validation split shares the training entity pool, so it could not see this
either. The course Q&A anticipates exactly this failure: asked about a
finetuning loss "constantly reducing to zero", the answer is *"your model might
have overfit on to one exact kind of template"*. Our `rationale` runs reached a
validation loss of 0.015 and scored 0.18 on test, which is that diagnosis with
numbers attached. Three epochs over 40,000 examples from ten templates drove the model
onto the template distribution; the extra 1,250 steps bought nothing and cost
almost everything.

#### Where the extra accuracy came from

Not from better reasoning. Splitting by what the answer is:

| Family group | Model H aggr -> gentle | Model L aggr -> gentle |
| :--- | :--- | :--- |
| Answer is a relation word | 0.5335 -> 0.5155 | 0.5360 -> 0.5070 |
| Answer is an entity name | **0.0540 -> 0.1550** | **0.0000 -> 0.0487** |

Relation-word accuracy is flat, in fact marginally *down*. **Every bit of the
gain is on the families that require naming an entity the model has never
seen** -- Hindi `val2_which` goes 0.0000 -> 0.3100, `rel2_which` 0.2640 ->
0.3500, `rel3_extreme` 0.0500 -> 0.1580; Nepali's six name-answering families go
from uniformly 0.0000 to 0.02-0.10.

**This links the two failures into one.** §5.3 showed the aggressive models
could only ever emit a training-pool name. Naming an entity from the prompt is a
*language-modelling* act -- copying a token out of context -- and the aggressive
recipe destroyed the language model. The gentle recipe keeps 96% of it, and with
it enough of the copying ability to sometimes name a held-out entity. Forgetting
and the memorisation failure are not two findings; they are one mechanism seen
from two sides.

**How far that claim carries.** Within each model the relationship is monotone
(Hindi 33% -> 96% retained gives 0.054 -> 0.155; Nepali 18% -> 96% gives 0.000
-> 0.049). *Across* models it is not: Nepali at 96% retention scores 0.0487 on
name-answering, slightly below Hindi at 33% retention on 0.0540. So language-model
retention predicts name-answering ability *within* a model and does not
transfer across the two. Four points, two of them confounded with language. This
is a mechanism consistent with the evidence, not an established dose-response.

**One family moved the other way.** `val2_relation` -- the only value-premise
family whose answer is a relation word -- fell in both models (0.4760 -> 0.4260
Hindi, 0.5360 -> 0.4020 Nepali). It is the family that most requires comparing
two numerals, and it is the one thing the longer run did better. §7.3 returns to
this.

**Which configuration the report treats as the result.** The gentle one. It is
better on every measured axis and cheaper to train. The aggressive run is
retained and reported in full because the comparison is the finding -- and
because §4.2's reasoning for the aggressive settings was explicit and wrong,
which is more useful on the record than silently replaced.

**What was not re-run at the gentle setting.** The control split (`test_seen`),
the attention comparison, the `rationale` style, and the second seed. Their
numbers in §5.2-5.4 and §6 all describe the aggressive checkpoints. The
memorisation gap, the seed range and every attention statistic should therefore
be read as properties of that configuration; §9 records this as a limitation.

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

512 prompts, 19-65 tokens wide, 59 unpadded batches, **18,689 query positions
for each checkpoint** -- identical counts, which is the guarantee that both saw
exactly the same input. Full detail in
`{hindi,nepali}/eval/attention_reasoning.json`.

| | Model H | Model L |
| :--- | ---: | ---: |
| Mean attention distance, pretrained | 8.76 | 8.20 |
| Mean attention distance, finetuned | **7.70** | **7.70** |
| Layers where distance fell | **8 of 8** | **8 of 8** |
| Heads that changed role | 21 / 64 | 20 / 64 |

Per-layer, Model H:

| Layer | distance pre -> ft | normalised entropy pre -> ft |
| ---: | :--- | :--- |
| 0 | 9.01 -> 8.90 (-0.10) | 0.910 -> 0.933 (+0.023) |
| 1 | 10.02 -> 9.34 (-0.68) | 0.806 -> 0.832 (+0.026) |
| 2 | 8.86 -> 7.53 (-1.33) | 0.749 -> 0.742 (-0.007) |
| 3 | 4.66 -> 4.14 (-0.52) | 0.684 -> 0.599 (-0.085) |
| 4 | 4.95 -> 4.19 (-0.76) | 0.550 -> 0.464 (-0.086) |
| 5 | 10.88 -> 8.14 (**-2.74**) | 0.489 -> 0.518 (+0.029) |
| 6 | 11.16 -> 9.63 (-1.52) | 0.578 -> 0.541 (-0.037) |
| 7 | 10.53 -> 9.73 (-0.81) | 0.662 -> 0.594 (-0.068) |

Head-role transitions (Phase 2's classifier):

- **Model H**: `local -> local` 33, `content-based -> local` **10**,
  `previous-token -> previous-token` 7, `sink -> local` 4,
  `sink -> content-based` 4, `content-based -> content-based` 3, three singletons.
- **Model L**: `local -> local` 31, `previous-token -> previous-token` 9,
  `content-based -> local` **9**, `sink -> content-based` 5,
  `local -> content-based` 3, `content-based -> content-based` 2, five singletons.

Heatmaps: `report/figures/{hindi,nepali}_reasoning_attention_direct_layer{0,7}.png`,
drawn on a `rel4_pairwise` prompt (49 tokens, 2 hops, answer `कम`).

### 6.3 Required discussion -- local versus long-range, and head specialisation

**Finetuning made attention narrower and more local, in both models, at every
layer.** Mean distance fell at all eight layers in both, the largest single
movement being Model H's layer 5 (10.88 -> 8.14). Normalised entropy fell in the
middle and late layers -- most at layers 3 and 4 (-0.085, -0.086) -- so the
heads are not merely looking closer, they are concentrating on fewer positions.

The role transitions say the same thing in a different vocabulary. The dominant
non-identity transition in both models is **`content-based -> local`** (10 in H,
9 in L): heads that had been selecting on *what the tokens are* moved to
selecting on *where they are*. Nothing moved the other way in any quantity --
`local -> content-based` accounts for 1 head in H and 3 in L.

**§6.3 pre-registered four readings before the run. The data picked the second:**

> *"Mean distance decreases, entropy falls -- the model sharpened onto local
> template structure (the fixed `प्रश्न:` / `उत्तर:` frame) rather than onto
> content. Combined with a high format rate and a low exact match, this would be
> direct evidence that finetuning taught format rather than reasoning."*

All three conditions hold: distance down at every layer, entropy down in the
middle and late layers, format rate 0.96-1.00 against exact match 0.21-0.25.
The reading that would have indicated genuine chaining -- *"mean distance
increases in late layers... the model learned to reach back to the premises"* --
is contradicted: layer 7 moved the other way in both models.

**Head specialisation was largely preserved, as expected.** 21/64 and 20/64
heads changed label, so roughly two-thirds kept their role. §6.3 predicted this
and said stability would be the unsurprising outcome: 1,875 steps at 1e-4 is a
small perturbation next to 14,329 at 3e-4. It is worth noting that this
prediction held for *attention structure* while the equivalent prediction for
*language-modelling ability* failed badly (§5.5) -- the same update left most
heads doing the same job and still destroyed two-thirds of the model's grip on
its language.

**The `previous-token` heads are the most stable group**: 7/7 in Model H and 9/9
in Model L kept their role, with none acquired or lost. Phase 2 identified these
as structural, offset-fixed heads; they are unaffected by what the model is
being asked to do, which is consistent with that reading.

**Both models converge on 7.70.** Two models sharing no data, no vocabulary and
no weights land on the same mean attention distance after finetuning, to three
significant figures, having started 0.56 apart. Phase 2 found the same
near-identity on corpus windows and drew the same conclusion: these statistics
are dominated by the architecture and the objective rather than by the corpus.
With two observations this is an observation, not a result.

**Caveat, restated.** Mean distance scales with sequence length, and these
prompts average ~37 tokens against Phase 2's 512-token windows. The numbers
above are comparable pretrained-against-finetuned only, never against §6 of the
Phase 2 report.

## 7. The Four Required Questions

### 7.1 How did data scale and quality differ between Model H and Model L?

**Answered in full; this does not depend on any outstanding run.**

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

**Both answered.**

The Phase 2 language-modelling comparison was genuinely ambiguous, and the
ambiguity was the finding. Hindi won per-token perplexity by 14.4% (26.80
against 30.65); Nepali won bits-per-character by 14.0% (1.1964 against 1.3636).
Both were true, because Nepali's tokenizer packs 18.6% more text into each
token -- so predicting a Nepali token is a strictly harder problem, which
perplexity charges it for and bits-per-character credits it for.

**Reasoning accuracy resolves that ambiguity, because exact match is
tokenizer-independent.** A correct answer is correct regardless of how many
tokens it cost to encode, so a gap here cannot be attributed to fertility. The
result:

| | Model H | Model L |
| :--- | ---: | ---: |
| Exact match, 3 epochs @ 1e-4 | **0.2458** | 0.2144 |
| Exact match, 1 epoch @ 3e-5 (§5.6) | **0.2992** | 0.2320 |
| Exact match, seen entities (aggressive) | **0.6226** | 0.4542 |
| Seed range (aggressive) | 0.0126 | 0.0026 |
| LM retained, aggressive | **33%** | 18% |
| LM retained, gentle | 96% | 96% |

**Model H is ahead on reasoning under both recipes, and it is the more robust of
the two.** It scores 14.7% higher on unseen entities at the aggressive setting
and 29% higher at the gentle one, 37% higher on seen entities, and it kept
nearly twice as much of its language modelling through the aggressive finetune
(33% against 18%). At the gentle setting both retain 96%, so that particular gap
is a property of how hard the models were pushed rather than of the models.

So the two tiers rank *consistently* on reasoning where they ranked
*contradictorily* on language modelling. That is the cleanest statement this
project can make about the resource gap: once the metric stops depending on the
tokenizer, the higher-resource model is better -- modestly on the task, clearly
on robustness.

**With the honest qualifier.** The reasoning gap on unseen entities (0.0314) is
about 2.5x Model H's own seed range (0.0126). The direction is supported; a
precise magnitude is not. And both models are poor in absolute terms (§5.3), so
this is a comparison between two weak models, not a demonstration that either
works.

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

Factors 2 and 3 point in **opposite directions**, and §3.5 registered a test to
decide which dominates: compare Nepali's relative accuracy on the four
value-premise families against the six relation-premise ones. The measurement:

| Family group | Model H | Model L | L / H |
| :--- | ---: | ---: | ---: |
| Value premises (quantities stated) | 0.1190 | 0.1340 | **1.126** |
| Relation premises (inequalities stated) | 0.3303 | 0.2680 | 0.811 |

**The prediction holds in direction.** Nepali does relatively *better* exactly
where numerals must be read — 12.6% ahead of Hindi — and relatively worse
everywhere else, by 19%. The numeral-segmentation advantage measured in §3.5
(1.46 against 2.04 tokens per quantity) appears to do real work.

**But the evidence is thin, and saying so matters more than the result.** Three
of the four value-premise families score exactly 0.0000 in both languages,
because their answers are entity names and neither model can produce an unseen
one (§5.3). The whole value-premise average therefore rests on `val2_relation`
alone — 0.4760 against 0.5360 — which is one family, and the only value family
whose answer is a relation word rather than a name. So the comparison is
confounded with answer type, the very variable that dominates everything else in
these results.

**And at the gentler setting the sign reverses.** Repeating the same test on the
§5.6 checkpoints:

| Recipe | value-premise L/H | relation-premise L/H | numeral advantage |
| :--- | ---: | ---: | :--- |
| 3 epochs @ 1e-4 | 1.126 | 0.811 | supported |
| 1 epoch @ 3e-5 | **0.629** | 0.830 | **contradicted** |

**Verdict: not established.** A conclusion that flips sign when the learning
rate changes is a property of the training recipe, not of the tokenizer. The
entity-naming failure is large enough to swamp any numeral effect, and at the
gentle setting -- where name-answering partly works -- Nepali does relatively
*worse* on the families that state quantities, not better.

The one consistent trace of a numeral effect is `val2_relation`, the single
family that both states quantities and asks for a relation word. It is also the
only family the *longer* run did better in both languages (0.4760 -> 0.4260
Hindi, 0.5360 -> 0.4020 Nepali, §5.6). Comparing numerals appears to be the one
thing that benefits from more training, which is consistent with it being the
hardest thing in the corpus -- but it is one family, and it is not enough.

**What a clean test would need**, and this is the most concrete improvement
available to the dataset: for every family that states quantities, a variant
whose answer is a relation word rather than a name. That would separate numeral
handling from entity naming instead of confounding the two, and the templates
already exist to build it.



### 7.4 What evidence explains the observed differences?

**Answered.** The table states each claim with its evidence and how far that
evidence actually carries.

| Claim | Evidence | Strength |
| :--- | :--- | :--- |
| The two models differ in data, not capacity | Identical architecture, 25,379,840 parameters each, identical hyperparameters | **Controlled by construction** |
| Both models memorised entity names rather than learning the relation | Every wrong answer on a name-answering family is a *training-pool* name; 0.6226 on seen entities against 0.2456 on unseen | **Direct, from the decoded generations** |
| Finetuning taught format, not reasoning | Pretrained + 4 examples: format rate 0.9998, exact match 0.0000. Finetuned: format 0.96-1.00, exact match 0.21-0.25 | **Direct** |
| Attention moved to local template structure | Distance fell at 8/8 layers in both models; `content-based -> local` the dominant role shift | **Direct, same batches both checkpoints** |
| Finetuning destroyed most language modelling | Perplexity 20.19 -> 1,115.21 (H), 23.77 -> 2,801.61 (L), same split, same slice | **Direct** |
| Results are not seed artefacts | Two seeds: range 0.0126 (H), 0.0026 (L) | **Strong for stability**, weak as an interval |
| Perplexity favoured H in Phase 2 for tokenizer reasons | Fertility 3.481 vs 4.126 chars/token, measured on the split files | Strong, directly measured |
| Nepali's corpus is more homogeneous | Manual share 21.20% vs 26.87%; `sangraha` 77.5% vs 71.6% of validation docs; 21.2% vs 10.6% out-of-domain fertility loss | Strong as measurement, **correlational** as explanation |
| Nepali's numeral advantage helps it where quantities are read | L/H ratio 1.126 on value premises against 0.811 on relation premises | **Suggestive only** -- rests on one family (§7.3) |
| Model H is better at the task than Model L | 0.2458 vs 0.2144 unseen; consistent across seen entities and forgetting | Supported in direction, **magnitude not established** (2.5x seed range) |

**Three things the design contributed that the numbers alone would not have.**
Disjoint entity pools turned an unexplained 0.25 into a diagnosed memorisation
failure. The three test sets separated "did not learn" from "memorised names"
from "memorised wording", and the answer was unambiguously the middle one.
Reporting format rate alongside accuracy is what made the few-shot result --
99.98% well-formed, 0.00% correct -- legible at all.

**The honest limit.** With two models there are two data points, and corpus
size, provenance mix, fertility and domain balance all differ at once. Nothing
here attributes an H-vs-L gap to a single cause. The closest thing to a
controlled comparison in this phase is *within* one model: the per-family and
per-test-set breakdowns vary answer type, entity familiarity and wording while
holding corpus, tokenizer and weights fixed. Those comparisons are where the
strong claims above come from; the cross-model ones are weaker and are labelled
as such.

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
7. **Checkpoint selection rewarded memorisation.** The validation split shares
   the training entity pool, so selecting on validation exact match could not
   see the dominant failure -- and picked the `rationale` checkpoints, which
   scored 0.924 on validation and 0.18 on test (§5.1). A validation split drawn
   from a *third*, disjoint entity pool would fix this; the inventories have the
   names for it. Not changed after the fact, because it would invalidate every
   number reported here.
8. **The gentler configuration was not fully evaluated.** §5.6 found 1 epoch at
   3e-5 better on every axis measured, but only the main test split and the
   forgetting check were re-run on it. The control split, the attention
   comparison, the `rationale` style and the second seed all describe the
   aggressive checkpoints, so the memorisation gap, the seed range and every
   number in §6 belong to a configuration the report itself recommends against.
   Re-running those four is about 20 GPU-minutes and is the first thing to do
   with more time.
9. **The forgetting measurement is a 2M-token slice**, not the full 26M
   validation split (§5.5). That is ample to separate "unchanged" from
   "degraded", but a small real drift and sampling noise are not distinguishable
   at that budget, and the figures are not directly comparable to Phase 2's
   full-split numbers.

---

## 10. Conclusion

Phase 3 finetuned two independent 25.4M-parameter models on synthetic
comparative-reasoning corpora in Hindi and Nepali, and the honest summary is
that **both models learned the shape of the task and little of the task itself**.

**What was measured.** Finetuning takes exact match from 0.0000 to 0.2992
(Model H) and 0.2320 (Model L) on held-out entities, with format compliance at
0.99-1.00. Accuracy splits cleanly on one variable that is neither reasoning
depth nor premise form: where the answer is one of three relation words both
models reach 0.45-0.67 against a 0.33 chance baseline, and where the answer is
an entity name they reach 0.02-0.35. Attention moved closer in at all eight
layers of both models, with `content-based -> local` the dominant head-role
shift. Model H leads Model L on every axis, by 29% on accuracy at the gentle
setting and by a factor of two on language-model retention at the aggressive
one.

**The three design decisions that made any of this legible.** Disjoint entity
pools, which turned an unexplained 0.25 into a diagnosed memorisation failure.
Three test sets rather than one score, which separated "did not learn" from
"memorised the names" from "memorised the wording" and identified the middle
one. And reporting format rate beside accuracy, without which the sharpest
number in the report -- a pretrained model that formats 99.98% of answers
correctly and gets 0.00% right -- would have been invisible.

**Two predictions this report made and got wrong**, both left on the record
rather than revised:

- §4.2 specified 3 epochs at a peak 1e-4 and §4.1 called it "a small
  perturbation". It destroyed 67% and 82% of what pretraining had taught the two
  models, and a third of the training at a third the learning rate was better on
  every axis measured (§5.6).
- §3.3 justified drawing the validation split from the training entity pool. That
  made checkpoint selection structurally blind to the dominant failure and
  selected the `rationale` runs, which scored 0.92 on validation and 0.18 on
  test (§5.1).

**The finding that connects everything.** Naming an entity from the prompt is a
language-modelling act -- copying a token out of context. The aggressive recipe
destroyed the language model and with it the copying, so those families scored
zero; the gentle recipe kept 96% of it and they partly work, which is where all
of its accuracy advantage comes from. Catastrophic forgetting and the
memorisation failure are one mechanism observed from two directions, not two
results.

**What the project as a whole shows about the two resource tiers.** Phase 2's
comparison was ambiguous by construction: Hindi won per-token perplexity, Nepali
won bits-per-character, and the difference was tokenizer fertility rather than
model quality. Exact match is the first metric in this project that does not
depend on the tokenizer, and it ranks the two models consistently, with Hindi
ahead. That is the clearest statement available here -- and it is a comparison
between two weak models on a narrow task, from two runs with no confidence
intervals and every corpus factor varying at once. The instrumentation is
trustworthy; the conclusions it supports are correspondingly narrow, and §9 says
where each one stops.
