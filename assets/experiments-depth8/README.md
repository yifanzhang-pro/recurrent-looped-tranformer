# Depth-eight experiment figures

These figures accompany the September 15, 2026 revision of [Recurrent Looped Transformer](../../Recurrent_Looped_Transformer.pdf).
Source manuscript revision: `b1f135c`.
The frozen snapshot was collected on September 15, 2026, 16:11:47–16:12:10 UTC.
Of 48 runs, 21 had reached 2,000 steps and 27 were unfinished.

All six models have eight logical layers and width 512, with global batch 512.
Parity uses initialization seeds 42, 43 and 44 on a fixed data stream and validation set.
Every parity aggregate requires all three seeds at the same step and reports mean ± sample SD (ddof=1).
Other tasks use seed 42.
Results use held-out validation sets at training lengths.

| Figure | PNG | PDF |
| --- | --- | --- |
| Validation accuracy | [PNG](validation-curves.png) | [PDF](validation-curves.pdf) |
| Parity at step 500 | [PNG](parity-errorbars.png) | [PDF](parity-errorbars.pdf) |
| Individual parity seeds | [PNG](parity-individual-seeds.png) | [PDF](parity-individual-seeds.pdf) |
| Training loss | [PNG](training-loss.png) | [PDF](training-loss.pdf) |

Addition requires an exact greedy answer and EOS.
Formal tasks report final-answer or final-state accuracy; S5 prefix accuracy is a separate metric.
The shared steps are addition 1,000, parity 500, flat and bracketed mod-5 800, and swaps and standard S5 1,000.
Training-loss curves show unsmoothed pre-update global-batch loss, with a display floor of 1e-8 on logarithmic axes.
