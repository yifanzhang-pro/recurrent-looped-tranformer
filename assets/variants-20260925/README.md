# Feedback-variant and sixteen-layer parity figures

Figure labels are synchronized with manuscript commit [d6a8ffe](https://github.com/yifanzhang-pro/Recurrent-Looped-Transformer-Overleaf/commit/d6a8ffecf777794c407e906ea376aeafdb08dd9e).
They come from the completed September 25, 2026 snapshot.

RLT-0 and RLT-2 chunk4 were trained at every eight-layer split (4+4 through 8+0) with seeds 42, 43 and 44, using the data and optimizer of the RLT-1 runs.
Mod 5 trains for 5,000 steps; addition, parity and S5 train for 2,000.
The RLT-1 and GPT Layer 8 controls use the same seeds, including 5,000-step mod-5 runs.
Every curve shows mean ± sample SD (n = 3, ddof = 1).
Length generalization uses each run's checkpoint with the lowest in-distribution validation loss, taking the earliest step on ties, and 1,024 shared examples per length.

The sixteen-layer parity comparison uses seed 42 and global batch 1,024; all ten runs completed 2,000 steps.
It compares each run's best in-distribution checkpoint with its step-2,000 checkpoint on the same examples.

| Figure | PNG | PDF | SVG |
| --- | --- | --- | --- |
| Validation at 4+4, all variants | [PNG](variants-validation-overview.png) | [PDF](variants-validation-overview.pdf) | [SVG](variants-validation-overview.svg) |
| Parity length generalization | [PNG](parity-generalization.png) | [PDF](parity-generalization.pdf) | [SVG](parity-generalization.svg) |
| Swaps-S5 length generalization | [PNG](s5-swaps-generalization.png) | [PDF](s5-swaps-generalization.pdf) | [SVG](s5-swaps-generalization.svg) |
| Standard S5 length generalization | [PNG](s5-standard-generalization.png) | [PDF](s5-standard-generalization.pdf) | [SVG](s5-standard-generalization.svg) |
| Flat mod 5: validation | [PNG](mod5-no-brackets-validation.png) | [PDF](mod5-no-brackets-validation.pdf) | [SVG](mod5-no-brackets-validation.svg) |
| Flat mod 5: length generalization | [PNG](mod5-no-brackets-generalization.png) | [PDF](mod5-no-brackets-generalization.pdf) | [SVG](mod5-no-brackets-generalization.svg) |
| Bracketed mod 5: validation | [PNG](mod5-with-brackets-validation.png) | [PDF](mod5-with-brackets-validation.pdf) | [SVG](mod5-with-brackets-validation.svg) |
| Bracketed mod 5: length generalization | [PNG](mod5-with-brackets-generalization.png) | [PDF](mod5-with-brackets-generalization.pdf) | [SVG](mod5-with-brackets-generalization.svg) |
| Sixteen-layer parity: best versus step 2,000 | [PNG](parity-depth16-best-vs-step2000.png) | [PDF](parity-depth16-best-vs-step2000.pdf) | [SVG](parity-depth16-best-vs-step2000.svg) |

The three-seed addition feedback-scale ablation is in [`../addition-feedback-20261005/`](../addition-feedback-20261005/).
