# Depth-eight experiment figures

These figures accompany the September 17, 2026 paper revision.
Source manuscript commit: [94d36d1](https://github.com/yifanzhang-pro/Recurrent-Looped-Transformer-Overleaf/commit/94d36d1).
The public English and Chinese PDFs include Figure 1 on page one.

All 108 runs completed 2,000 steps: six architectures, six tasks and initialization seeds 42, 43, 44.
Training and held-out examples are fixed across initializations.
Every aggregate reports mean ± sample SD (n = 3, ddof = 1).
Addition uses teacher-forced answer-token accuracy, including answer formatting and EOS and excluding prompt/padding positions.
Formal-task primary metrics score the final label or state; S5 token accuracy scores each prefix state.

Length generalization selects each run's minimum in-distribution validation-loss checkpoint, with earliest-step tie breaking.
All models and seeds share each test set: 256 addition pairs or 1,024 formal-task sequences per length.
The study covers 846 model–seed–length combinations and 282 three-seed aggregates.
Error bars measure initialization variability on these shared examples.

| Figure | PNG | PDF |
| --- | --- | --- |
| generalization-depth08-primary | [PNG](generalization-depth08-primary.png) | [PDF](generalization-depth08-primary.pdf) |
| generalization-depth08-s5-metrics | [PNG](generalization-depth08-s5-metrics.png) | [PDF](generalization-depth08-s5-metrics.pdf) |
| generalization-depth08-token-accuracy | [PNG](generalization-depth08-token-accuracy.png) | [PDF](generalization-depth08-token-accuracy.pdf) |
| parity-errorbars | [PNG](parity-errorbars.png) | [PDF](parity-errorbars.pdf) |
| parity-individual-seeds | [PNG](parity-individual-seeds.png) | [PDF](parity-individual-seeds.pdf) |
| training-loss | [PNG](training-loss.png) | [PDF](training-loss.pdf) |
| validation-curves | [PNG](validation-curves.png) | [PDF](validation-curves.pdf) |
| validation-token-accuracy | [PNG](validation-token-accuracy.png) | [PDF](validation-token-accuracy.pdf) |

Parity error bars compare steps 500 and 2,000.
Training curves extend through step 2,000; loss curves are unsmoothed and measured before each optimizer update, with a logarithmic display floor of 1e-8.
Fixed-length token accuracy uses length 33 for flat mod-5 and 32 for the other formal tasks.
The sixteen-layer addition figures in the neighboring directory are a separate seed-42 snapshot with four unfinished runs; they are not part of the three-seed aggregates above.
