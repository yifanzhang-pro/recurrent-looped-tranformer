# Sixteen-layer and feedback-variant experiments

September 20, 2026 snapshot from manuscript commit [f3102f6](https://github.com/yifanzhang-pro/Recurrent-Looped-Transformer-Overleaf/commit/f3102f63b905235ed228e6dfaa58c96fa71d8c81). All new experiments use initialization seed 42. The [depth-eight three-seed study](../experiments-depth8/README.md) is unchanged.

## Protocol

- Sixteen-layer addition: all ten models completed 2,000 steps and select the step-2,000 checkpoint. Teacher-forced answer-token accuracy includes formatting and EOS. Each width uses 256 shared operand pairs.
- Sixteen-layer parity: nine models completed 2,000 steps. RLT-1 8+8 is ongoing through step 1,800 and selects step 1,400; the completed 9+7 run selects step 1,300. Each length uses 1,024 shared sequences. The dagger marks ongoing training.
- Mod-5: 40 of 42 runs completed 5,000 steps, contributing 360 test points. RLT-1 4+4 is ongoing on both tasks; its OOD curves are pending. Validation includes unfinished runs, marked with open endpoints. Each test length uses 1,024 shared examples.
- Checkpoints minimize ID validation loss, with earliest-step tie breaking. Test scores do not select checkpoints. These single-seed results have no across-seed error bars.
- CPU timings average steps 1,001–2,000 with four threads and FP32. They exclude validation, saving and logging; shared-node load varies. They do not measure GPU throughput or inference speed.

[Source protocol, data and verification](https://github.com/yifanzhang-pro/Recurrent-Looped-Transformer-Overleaf/tree/f3102f63b905235ed228e6dfaa58c96fa71d8c81/data/results-refresh-20260919) · [Numerical tables](https://github.com/yifanzhang-pro/Recurrent-Looped-Transformer-Overleaf/tree/f3102f63b905235ed228e6dfaa58c96fa71d8c81/data/results-refresh-20260919/tables.md) · [Figure builder](https://github.com/yifanzhang-pro/Recurrent-Looped-Transformer-Overleaf/tree/f3102f63b905235ed228e6dfaa58c96fa71d8c81/scripts/plot_results_refresh.py)

## Figures

| Figure | PNG | SVG | PDF |
| --- | --- | --- | --- |
| arithmetic-generalization-depth16-series | [PNG](arithmetic-generalization-depth16-series.png) | [SVG](arithmetic-generalization-depth16-series.svg) | [PDF](arithmetic-generalization-depth16-series.pdf) |
| arithmetic-generalization-rlt44-vs88 | [PNG](arithmetic-generalization-rlt44-vs88.png) | [SVG](arithmetic-generalization-rlt44-vs88.svg) | [PDF](arithmetic-generalization-rlt44-vs88.pdf) |
| mod5-no-brackets-generalization | [PNG](mod5-no-brackets-generalization.png) | [SVG](mod5-no-brackets-generalization.svg) | [PDF](mod5-no-brackets-generalization.pdf) |
| mod5-no-brackets-validation | [PNG](mod5-no-brackets-validation.png) | [SVG](mod5-no-brackets-validation.svg) | [PDF](mod5-no-brackets-validation.pdf) |
| mod5-with-brackets-generalization | [PNG](mod5-with-brackets-generalization.png) | [SVG](mod5-with-brackets-generalization.svg) | [PDF](mod5-with-brackets-generalization.pdf) |
| mod5-with-brackets-validation | [PNG](mod5-with-brackets-validation.png) | [SVG](mod5-with-brackets-validation.svg) | [PDF](mod5-with-brackets-validation.pdf) |
| parity-generalization-depth16-series | [PNG](parity-generalization-depth16-series.png) | [SVG](parity-generalization-depth16-series.svg) | [PDF](parity-generalization-depth16-series.pdf) |

All figures are copied unchanged from the manuscript. [Source hashes](source.json) identify the exact files. The older `assets/arithmetic-generalization-depth16/` paths also serve the completed addition results.
