### No-positional-embedding ablation — hindi

Learning-rate schedules match across 716 shared steps (max relative difference 0.0000).

| At step 14,329 | Learned positions | No positions |
|:--|--:|--:|
| Validation loss (nats) | 3.0053 | 3.0469 |
| Validation perplexity | 20.19 | 21.05 |
| Gap | — | +0.0417 nats (1.04x perplexity) |

| Attention | Learned positions | No positions |
|:--|--:|--:|
| Strongest previous-token head | 0.96 | 0.88 |
| Heads with prev-token mass > 0.25 | 6 | 5 |
| Heads with mean distance < 10 | 6 | 12 |
| Mean attention distance | 65.6 | 71.3 |
| Mean normalised entropy | 0.643 | 0.689 |
