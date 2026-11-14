# Sample output: probe

Probe: the untrained model's answers on SFT calls. Its hidden states are in reads/probe_hidden_*.npz

Call 60332. The model read the prompt in [examples/prompts/60332_noblock.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/60332_noblock.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: up
Pricing: flat
Margins: up
Guidance: none
Candor: direct
Next EPS: up
Action: BUY
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 0.9231 | 0.9279 | yes |
| Demand | flat | 0.0669 | 0.0672 |  |
| Demand | down | 0.0048 | 0.0049 |  |
| Pricing | up | 0.1820 | 0.1823 |  |
| Pricing | flat | 0.8158 | 0.8169 | yes |
| Pricing | down | 8.43e-04 | 0.0008 |  |
| Margins | up | 0.7128 | 0.7132 | yes |
| Margins | flat | 0.2622 | 0.2624 |  |
| Margins | down | 0.0244 | 0.0244 |  |
| Guidance | raised | 0.1445 | 0.1450 |  |
| Guidance | held | 0.3467 | 0.3479 |  |
| Guidance | cut | 8.07e-04 | 0.0008 |  |
| Guidance | none | 0.5044 | 0.5062 | yes |
| Candor | direct | 0.9795 | 0.9797 | yes |
| Candor | evasive | 0.0203 | 0.0203 |  |
| Next EPS | up | 0.8717 | 0.8933 | yes |
| Next EPS | down | 0.1041 | 0.1067 |  |
| Action | BUY | 0.8905 | 0.8933 | yes |
| Action | AVOID | 0.1064 | 0.1067 |  |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.8717 ÷ (0.8717 + 0.1041) = **0.8933**
- **p(BUY)** = 0.8905 ÷ (0.8905 + 0.1064) = **0.8933**. This is the stock score.

The published file has p(BUY) = 0.8933 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `up` 0.9231, `flat` 0.0669, `down` 0.0048, `raised` 0.0038, `held` 5.79e-04 |
| Pricing | `flat` 0.8158, `up` 0.1820, `down` 8.43e-04, `held` 5.79e-04, `none` 2.13e-04 |
| Margins | `up` 0.7128, `flat` 0.2622, `down` 0.0244, `held` 1.64e-04, `stable` 9.36e-05 |
| Guidance | `none` 0.5044, `held` 0.3467, `raised` 0.1445, `cut` 8.07e-04, `None` 3.16e-04 |
| Candor | `direct` 0.9795, `ev` 0.0203, `indirect` 1.36e-05, `e` 6.41e-06, `clear` 6.02e-06 |
| Next EPS | `up` 0.8717, `down` 0.1041, `flat` 0.0124, `none` 0.0052, `unknown` 0.0052 |
| Action | `BUY` 0.8905, `A` 0.1064, `AV` 9.20e-04, `HOLD` 4.35e-04, `buy` 4.08e-04 |
