# Recurrent Looped Transformer

[![Paper](https://img.shields.io/badge/Paper-b31b1b.svg)](./Recurrent_Looped_Transformer.pdf)
[![Website](https://img.shields.io/badge/Project-Website-blue)](https://yifanzhang-pro.github.io/recurrent-looped-tranformer/)

### Recurrent computation across prompt and response

**Recurrent Looped Transformer (RLT-1)** passes the decoder's final hidden state to the next token, together with that token's causal encoder representation.
The decoder reads encoder-derived global KV memory and maintains a sliding-window attention (SWA) cache at every layer.
The same update runs over prompt and response tokens.

**Authors:** [Yifan Zhang](https://yifzhang.com), Jichen Feng, Shihan Qin

**Report:** September 12, 2026 · **Updated:** September 17, 2026

[[Paper](./Recurrent_Looped_Transformer.pdf)] [[中文论文](./Recurrent_Looped_Transformer_ZH.pdf)] [[Project website](https://yifanzhang-pro.github.io/recurrent-looped-tranformer/)] [[Experiments](#depth-eight-experiments)]

![RLT recurrence across the last prompt tokens and the first response token.](figure1.png)

## Architecture

For token $x_t$, let $e_t$ be its causal encoder representation and $M_{\le t}$ the encoder-derived global KV memory.
The decoder state includes both the recurrent output and layerwise SWA KV:

```math
H_t=(s_t,C_t^D),\qquad H_0=(s_\star,\varnothing).
```

```math
(s_t,C_t^D)=D_\phi\!\left(\mathrm{Merge}(e_t,s_{t-1});M_{\le t},C_{t-1}^D,t\right).
```

- **Global context:** encoder outputs are projected into cached KV; cross-attention reads positions up to the current token. With one memory group ($G=1$), all decoder layers read the same projected KV using their own queries.
- **Local memory:** each decoder layer projects its own SWA KV. A window of $W$ includes the current token and retains up to $W-1$ past entries for the next update.
- **Hidden-state feedback:** the previous final decoder output enters the next token's gated merge. The state continues across the prompt–response boundary.

![Detailed RLT architecture showing the encoder, gated merge, global memory, per-layer SWA and recurrent feedback.](assets/architecture-detail.png)

[Architecture PDF](assets/architecture-detail.pdf)

After $t$ tokens, the recurrent path traverses $tL_D$ decoder blocks while the number of blocks evaluated per token stays fixed.
Compatible encoder and decoder attention and FFN weights can be shared; a tied 48+48 layout illustrates this option in the report.
The experiments below use untied eight-layer layouts.

### RLT-0: no hidden-state feedback

**RLT-0** removes the previous-output feedback path, learned initial state, state normalization, gated merge, and feedback projection.
The decoder receives $z_t^0=e_t$ directly and retains global cross-attention and layerwise SWA.
Known tokens can run in parallel within each decoder layer during training and prefill; generation proceeds one token at a time.
The 4+4 control is implemented for matched ablations and has no results in the tables below.

![RLT without hidden-state feedback: encoder outputs feed the decoder directly, with global KV and per-layer SWA retained.](assets/architecture-no-feedback.png)

[Control architecture PDF](assets/architecture-no-feedback.pdf)

### RLT-2: chunk-level feedback

RLT-2 is a proposed extension that holds the feedback state fixed within a chunk and updates it from the last decoder output at a complete chunk boundary.
Known positions can run together within each decoder layer, using causal SWA and prefix-restricted encoder memory.
Chunk boundaries are anchored at BOS and continue across prompt, response and message boundaries; a partial chunk preserves the previous boundary state.
With chunk size B = 1, the equations recover RLT-1 at the same weights.
The paper specifies the algorithms and execution costs; the reported experiments contain no RLT-2 training or throughput results.

![RLT-0, RLT-1 and RLT-2 decoder schedules for known tokens.](assets/rlt2/chunk-schedule.png)

[Chunk architecture](assets/rlt2/architecture-chunk.png) · [Schedule PDF](assets/rlt2/chunk-schedule.pdf)

## Depth-eight experiments

The September 17, 2026 snapshot compares **RLT-1 4+4, 5+3, 6+2, 7+1, 8+0 and Transformer 8** on six algorithmic tasks.
All **108 runs** completed **2,000 optimizer steps**, using initialization seeds **42, 43 and 44 for every task**.
Training examples and held-out sets are fixed across initializations.
All aggregates report **mean ± sample standard deviation** (n = 3, ddof = 1).

Models use width 512, FFN width 1,365, four attention heads, global batch 512, microbatch 32 and the same AdamW schedule.
RLT-1 uses an SWA window of eight, one shared encoder-memory group, feedback scale 0.1 and TBPTT 128, which covers every training sequence and cuts no gradients here.
RLT-1 has 26.10–28.73M parameters; Transformer 8 has 25.31M.
The 8+0 variant has no decoder blocks but still applies the gated recurrent merge.

### Validation accuracy after 2,000 steps

Values are percentages, reported as mean ± sample SD across three initializations.
Each run has consumed 1,024,000 training examples.
Addition measures **teacher-forced answer-token accuracy**, including answer formatting and EOS, excluding prompt and padding positions.
Parity and mod-5 score the final label; S5 scores the final state.

| Task | RLT-1 4+4 | RLT-1 5+3 | RLT-1 6+2 | RLT-1 7+1 | RLT-1 8+0 | Transformer 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Addition | 100.00±0.00 | 100.00±0.00 | 100.00±0.00 | 100.00±0.00 | 100.00±0.00 | 100.00±0.00 |
| Parity | 100.00±0.00 | 100.00±0.00 | 100.00±0.00 | 100.00±0.00 | 98.83±1.92 | 94.84±3.43 |
| Mod 5, no brackets | 45.36±46.43 | 94.18±7.29 | 69.62±33.01 | 70.01±43.15 | 60.33±35.70 | 64.02±37.64 |
| Mod 5, brackets | 70.53±10.75 | 74.35±6.71 | 74.87±3.72 | 74.78±3.01 | 75.17±6.78 | 73.87±9.07 |
| S5, swaps | 100.00±0.00 | 100.00±0.00 | 100.00±0.00 | 99.35±0.81 | 99.61±0.68 | 99.09±0.23 |
| S5, standard | 0.78±0.68 | 2.47±1.26 | 2.21±1.48 | 2.08±0.98 | 1.82±1.19 | 0.52±0.23 |

![Validation accuracy on six tasks, mean and sample SD across three initialization seeds.](assets/experiments-depth8/validation-curves.png)

All curves extend through step 2,000. Bands show sample SD, clipped to the accuracy range. Standard S5 uses a narrower vertical scale.
[PDF](assets/experiments-depth8/validation-curves.pdf)

### Learning speed and initialization

At step 500, RLT-1 6+2 reaches **99.44 ± 0.98%** parity accuracy, compared with **48.48 ± 0.53%** for Transformer 8.
All three 6+2 seeds reach 100% by step 600.
At step 2,000, splits 4+4 through 7+1 reach 100% in all three seeds, while Transformer 8 reaches 94.84 ± 3.43%.

![Parity accuracy at steps 500 and 2,000, with individual seeds, means and sample SD.](assets/experiments-depth8/parity-errorbars.png)

Both panels use the same 768 validation examples. They correspond to 256,000 and 1,024,000 training examples per seed. SD whiskers can extend beyond 100%.
[PDF](assets/experiments-depth8/parity-errorbars.pdf)

Flat mod-5 varies strongly with initialization: RLT-1 5+3 reaches **94.18 ± 7.29%**, compared with **64.02 ± 37.64%** for Transformer 8.
The three 4+4 seeds score 17.58%, 19.53% and 98.96%.
Bracketed mod-5 model means are closer, spanning 70.53–75.17% for RLT-1 versus 73.87 ± 9.07% for the Transformer.
The generators differ in operator structure and label distribution; parentheses also occupy token positions.

### Length generalization

Each run selects its lowest in-distribution validation-loss checkpoint over the full training history, taking the earliest step on ties.
Test results do not enter selection.
At every task and length, all models and seeds receive the same 256 addition pairs or 1,024 formal-task sequences.
The evaluation covers **846 model–seed–length combinations**, summarized as 282 three-seed means and sample SDs.

- **Parity:** 5+3 and 7+1 retain **100% accuracy at 256 bits in every seed**; Transformer 8 reaches 50.07 ± 1.63%.
- **Swaps-S5:** at 512 operations, 4+4 reaches **55.70 ± 25.78% final-state accuracy** and **91.16 ± 6.09% prefix-token accuracy**, versus 0.85 ± 0.30% and 9.33 ± 0.11% for Transformer 8.
- **Addition:** 7+1 reaches 68.05 ± 3.64% teacher-forced token accuracy at nine digits per operand. At 32 digits, all model means fall to 14.89–16.84%.
- **Modular arithmetic:** flat mod-5 approaches its 20% uniform reference at length 255; bracketed mod-5 also loses accuracy at longer lengths.

![Length generalization on all six tasks with three-seed error bars.](assets/experiments-depth8/generalization-depth08-primary.png)

Gray regions mark training lengths. Addition uses teacher-forced answer tokens; other tasks use final labels or states. Error bars show sample SD across initialization seeds.
[PDF](assets/experiments-depth8/generalization-depth08-primary.pdf)

#### Accuracy at the longest tested length

All entries are percentages, mean ± sample SD across three initializations.
Addition lengths count digits per operand; formal-task lengths count input symbols or operations, excluding boundary markers.

| Task (test length) | RLT-1 4+4 | RLT-1 5+3 | RLT-1 6+2 | RLT-1 7+1 | RLT-1 8+0 | Transformer 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Addition (32) | 15.30±0.44 | 14.89±2.27 | 15.74±1.58 | 15.59±1.57 | 15.31±0.93 | 16.84±1.45 |
| Parity (256) | 66.76±28.78 | 100.00±0.00 | 84.05±27.63 | 100.00±0.00 | 68.91±27.39 | 50.07±1.63 |
| Mod 5, no brackets (255) | 18.00±0.39 | 20.57±0.62 | 21.42±0.49 | 19.34±0.54 | 20.44±0.91 | 20.35±2.01 |
| Mod 5, brackets (256) | 25.20±1.71 | 25.81±4.34 | 21.58±1.86 | 22.04±1.72 | 22.30±1.13 | 25.07±2.05 |
| S5, standard (512) | 1.24±0.30 | 1.43±0.60 | 1.30±0.31 | 0.88±0.20 | 0.72±0.31 | 0.81±0.06 |
| S5, swaps (512) | 55.70±25.78 | 34.86±6.10 | 22.14±20.95 | 0.85±0.06 | 0.85±0.20 | 0.85±0.30 |

![S5 length generalization under prefix-token, final-state and whole-sequence scoring.](assets/experiments-depth8/generalization-depth08-s5-metrics.png)

Whole-sequence accuracy requires every prefix prediction to be correct. Every model uses the same 1,024 sequences at each length. Each panel labels its vertical scale.
[PDF](assets/experiments-depth8/generalization-depth08-s5-metrics.pdf)

The preferred encoder–decoder split depends on the task.
These comparisons change feedback, attention structure, parameter count and compute together; matched RLT-0 runs are needed to isolate hidden-state feedback.
Hardware-throughput and RL performance have not been measured in this study.

### Supplementary results

[Training loss](assets/experiments-depth8/training-loss.png), [individual parity seeds](assets/experiments-depth8/parity-individual-seeds.png), [fixed-length token accuracy](assets/experiments-depth8/validation-token-accuracy.png), and [token-level length generalization](assets/experiments-depth8/generalization-depth08-token-accuracy.png) provide additional diagnostics.
Training loss is unsmoothed and measured before the optimizer update; all six tasks use three seeds.

The paper also reports a separate **seed-42 sixteen-layer addition study** from September 16, 2026, with four unfinished runs.
It evaluates greedy exact-answer accuracy and teacher-forced token accuracy on shared operand pairs.
All nine RLT-1 layouts produce zero exact answers at every tested width from nine to 32 digits; Transformer 16 produces 3/256 at nine digits and zero at longer widths.
See the [sixteen-layer figure](assets/arithmetic-generalization-depth16/arithmetic-generalization-depth16-series.png) and [4+4 versus 8+8 comparison](assets/arithmetic-generalization-depth16/arithmetic-generalization-rlt44-vs88.png) for checkpoint steps and unequal training budgets.

[Experiment figures and protocol](assets/experiments-depth8/README.md)

## One execution across training and inference

| Mode | Encoder | Decoder and gradients |
| --- | --- | --- |
| Prompt prefill | Causal batch over known tokens | Recur through every prompt token and construct decoder SWA KV |
| Generation | Incremental encoding | Sample from the preceding state, then consume the token exactly once |
| Pretraining | Causal batch | Full BPTT; supervise every valid next-token target |
| SFT | Causal batch | Full BPTT; supervise assistant targets while updating state on all context tokens |
| Current-policy RL replay | Rebuild under current weights | Reconstruct the full history, including SWA caches; evaluate actions before consuming them |

Exact current-policy replay rebuilds parameter-dependent caches after weight updates.
Full gradients pass through recurrent outputs, decoder KV and encoder memory; detaching them changes the gradient.
The report specifies the sampling and replay distributions used for importance weighting.

## Independent community experiments

Synthetic state-tracking results contributed by [@AradhyeAgarwal](https://x.com/AradhyeAgarwal), using a small implementation with approximately **79K parameters** and **3 seeds**. Training length is **32 operations**; evaluation extends to **128 operations (4× the training length)**, with **2,048 test programs per task and length**.

![Independent RLT state-tracking results at up to four times the training length, comparing RLT, GRU, Transformer, and token-only merge.](./assets/rlt-state-tracking-results.png)

Final-state accuracy by number of operations:

**Parity**

| Model | 16 operations | 32 operations (train length) | 64 operations (2×) | 128 operations (4×) |
| --- | ---: | ---: | ---: | ---: |
| RLT | ≈100% | ≈100% | ≈82% | 60.8% |
| Transformer | ≈98% | ≈72% | ≈50% | ≈48% |
| Token-only merge | ≈98% | ≈59% | ≈49% | ≈50% |

**Five-state transitions**

| Model | 16 operations | 32 operations (train length) | 64 operations (2×) | 128 operations (4×) |
| --- | ---: | ---: | ---: | ---: |
| RLT | ≈100% | ≈100% | ≈49% | 20.7% |
| Transformer | ≈54% | ≈24% | ≈20% | ≈21% |
| Token-only merge | ≈50% | ≈23% | ≈20% | ≈20% |

Values marked ≈ are approximate readings from the original figure; exact values are not labeled. RLT's 128-operation values are taken from the figure's numeric labels.

RLT fits both tasks at the training length, but accuracy declines on longer sequences. Chance accuracy is 50% for parity and 20% for five-state transitions. Points in the original figure show means across seeds; whiskers show seed minima and maxima. Parameter and data budgets were matched; FLOPs were not. These community results use a separate implementation and evaluation protocol from the depth-eight snapshot above.


## Resources

- [Updated paper](./Recurrent_Looped_Transformer.pdf)
- [中文论文](./Recurrent_Looped_Transformer_ZH.pdf)
- [Project website](https://yifanzhang-pro.github.io/recurrent-looped-tranformer/)
- [Experiment figures](assets/experiments-depth8/README.md)
- [Prefill–decode kernel mismatch note](https://github.com/yifanzhang-pro/Pretraining-RL-Science/blob/master/Prefill_Decode_Kernel_Mismatch.pdf)

## Citation

```bibtex
@techreport{zhang2026recurrentlooped,
  title  = {Recurrent Looped Transformer},
  author = {Zhang, Yifan and Feng, Jichen and Qin, Shihan},
  year   = {2026},
  month  = sep,
  url    = {https://github.com/yifanzhang-pro/recurrent-looped-tranformer}
}
```

## License

Copyright 2026 Yifan Zhang. Licensed under the [Apache License 2.0](./LICENSE).
