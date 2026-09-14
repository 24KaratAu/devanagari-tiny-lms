### The three test sets

| Model | Style | Seen entities, seen wording | Unseen entities, seen wording | Unseen wording | Memorisation gap | Template gap |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Model H (Hindi) (seed 7) | direct | — | 0.2665 | 0.2331 | — | +0.0334 |
| Model H (Hindi) | direct | 0.6226 | 0.2456 | 0.2463 | +0.3770 | -0.0007 |
| Model H (Hindi) | rationale | — | 0.1836 | 0.1664 | — | +0.0172 |
| Model L (Nepali) (seed 7) | direct | — | 0.2211 | 0.2043 | — | +0.0168 |
| Model L (Nepali) | direct | 0.4542 | 0.2116 | 0.2232 | +0.2426 | -0.0116 |
| Model L (Nepali) | rationale | — | 0.2018 | 0.1911 | — | +0.0107 |

### Seed spread

| Model | Style | Seeds | Exact match per seed | Range |
| :--- | :--- | ---: | :--- | ---: |
| Model H (Hindi) | direct | 2 | 7: 0.2584, 42: 0.2458 | 0.0126 |
| Model L (Nepali) | direct | 2 | 7: 0.2170, 42: 0.2144 | 0.0026 |

### Pretrained vs finetuned exact match

| Model | Answer style | Setup | Exact match | Format rate | 0 hops | 1 hop | 2 hops |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Model H (Hindi) | direct | pretrained, zero-shot | **0.0000** | 0.0040 | 0.0000 | 0.0000 | 0.0000 |
| Model H (Hindi) | direct | pretrained, few-shot (k=4) | **0.0000** | 0.9998 | 0.0000 | 0.0000 | 0.0000 |
| Model H (Hindi) | direct | finetuned | **0.2458** | 0.9614 | 0.1480 | 0.3933 | 0.2690 |
| Model H (Hindi) | direct | finetuned | **0.2584** | 0.9760 | 0.1644 | 0.4093 | 0.2670 |
| Model H (Hindi) | rationale | finetuned | **0.1794** | 0.8686 | 0.0660 | 0.3547 | 0.2000 |
| Model L (Nepali) | direct | pretrained, zero-shot | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Model L (Nepali) | direct | pretrained, few-shot (k=4) | **0.0000** | 0.5374 | 0.0000 | 0.0000 | 0.0000 |
| Model L (Nepali) | direct | finetuned | **0.2144** | 1.0000 | 0.1072 | 0.3713 | 0.2470 |
| Model L (Nepali) | direct | finetuned | **0.2170** | 1.0000 | 0.1044 | 0.3787 | 0.2560 |
| Model L (Nepali) | rationale | finetuned | **0.1992** | 0.9716 | 0.0876 | 0.3473 | 0.2560 |

### Generalisation: unseen phrasing, and premise form

| Model | Style | Setup | Seen phrasing | Held-out phrasing | Gap | Value premises | Relation premises |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Model H (Hindi) | direct | pretrained, zero-shot | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 |
| Model H (Hindi) | direct | pretrained, few-shot | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 |
| Model H (Hindi) | direct | finetuned | 0.2456 | 0.2463 | -0.0007 | 0.1190 | 0.3303 |
| Model H (Hindi) | direct | finetuned | 0.2665 | 0.2331 | +0.0334 | 0.1450 | 0.3340 |
| Model H (Hindi) | rationale | finetuned | 0.1836 | 0.1664 | +0.0172 | 0.0795 | 0.2460 |
| Model L (Nepali) | direct | pretrained, zero-shot | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 |
| Model L (Nepali) | direct | pretrained, few-shot | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 |
| Model L (Nepali) | direct | finetuned | 0.2116 | 0.2232 | -0.0116 | 0.1340 | 0.2680 |
| Model L (Nepali) | direct | finetuned | 0.2211 | 0.2043 | +0.0168 | 0.1305 | 0.2747 |
| Model L (Nepali) | rationale | finetuned | 0.2018 | 0.1911 | +0.0107 | 0.1095 | 0.2590 |

### Per-family accuracy (finetuned)

| Family | H direct | H direct | H rationale | L direct | L direct | L rationale |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `eq3_relation` | 0.6660 | 0.6740 | 0.5580 | 0.6600 | 0.6760 | 0.6040 |
| `rel2_which` | 0.2640 | 0.2420 | 0.0120 | 0.0000 | 0.0000 | 0.0000 |
| `rel3_extreme` | 0.0500 | 0.0960 | 0.0020 | 0.0000 | 0.0000 | 0.0000 |
| `rel3_pairwise` | 0.4640 | 0.4580 | 0.5040 | 0.4540 | 0.4600 | 0.4380 |
| `rel4_extreme` | 0.0100 | 0.0140 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `rel4_pairwise` | 0.5280 | 0.5200 | 0.4000 | 0.4940 | 0.5120 | 0.5120 |
| `val2_relation` | 0.4760 | 0.5800 | 0.3140 | 0.5360 | 0.5220 | 0.4380 |
| `val2_which` | 0.0000 | 0.0000 | 0.0020 | 0.0000 | 0.0000 | 0.0000 |
| `val3_extreme` | 0.0000 | 0.0000 | 0.0020 | 0.0000 | 0.0000 | 0.0000 |
| `val4_rank` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Error analysis

**Model H (Hindi), direct answers** - most frequent errors (gold -> predicted):

- `अधिक -> कम` x 559
- `कम -> अधिक` x 276
- `सुधा -> उषा` x 91
- `संदूक -> अंजलि` x 52
- `बबीता -> कम्बल` x 50
- `कम -> बराबर` x 45
- `अधिक -> बराबर` x 43
- `तनुजा -> कम्बल` x 42

**Model H (Hindi), direct answers, seed 7** - most frequent errors (gold -> predicted):

- `अधिक -> कम` x 827
- `सुधा -> उषा` x 62
- `संदूक -> निशा` x 62
- `अर्चना -> बर्तन` x 43
- `बाल्टी -> गोपाल` x 42
- `तरुण -> नरेश` x 38
- `प्रवीण -> वंदना` x 30
- `सुधा -> सोहन` x 29

**Model H (Hindi), rationale answers** - most frequent errors (gold -> predicted):

- `कम -> अधिक` x 365
- `अधिक -> कम` x 300
- `अधिक -> <no answer>` x 133
- `कम -> <no answer>` x 126
- `अर्चना -> बर्तन` x 58
- `बराबर -> अधिक` x 53
- `अधिक -> बराबर` x 51
- `कम -> बराबर` x 48

**Model L (Nepali), direct answers** - most frequent errors (gold -> predicted):

- `कम -> बढी` x 567
- `बढी -> कम` x 323
- `रञ्जना -> रीता` x 97
- `कुचो -> कुर्सी` x 59
- `युवराज -> बाबुराम` x 56
- `बाकस -> बटुको` x 48
- `अस्मिता -> रीता` x 48
- `हेमन्त -> रीता` x 44

**Model L (Nepali), direct answers, seed 7** - most frequent errors (gold -> predicted):

- `बढी -> कम` x 820
- `रञ्जना -> रीता` x 98
- `बाकस -> भाँडा` x 55
- `युवराज -> रमेश` x 47
- `कुसुम -> श्याम` x 46
- `उर्मिला -> रीता` x 43
- `जीवन -> रीता` x 40
- `हेमन्त -> रीता` x 39

**Model L (Nepali), rationale answers** - most frequent errors (gold -> predicted):

- `बढी -> कम` x 643
- `कम -> बढी` x 202
- `रञ्जना -> रीता` x 75
- `बाल्टिन -> डोरी` x 49
- `साइकल -> डोरी` x 45
- `बराबर -> कम` x 42
- `बराबर -> बढी` x 31
- `ऐना -> शान्ति` x 29

