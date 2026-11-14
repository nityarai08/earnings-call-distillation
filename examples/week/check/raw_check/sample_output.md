# Sample output: raw_check

Call only setup on the check calls

Call 62273. The model read the prompt in [examples/prompts/62273_noblock_noreaction.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/62273_noblock_noreaction.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: up
Pricing: flat
Margins: up
Guidance: raised
Candor: direct
Next EPS: down
Action: AVOID
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 0.4775 | 0.5123 | yes |
| Demand | flat | 0.1991 | 0.2135 |  |
| Demand | down | 0.2556 | 0.2742 |  |
| Pricing | up | 0.0322 | 0.0323 |  |
| Pricing | flat | 0.9410 | 0.9426 | yes |
| Pricing | down | 0.0251 | 0.0251 |  |
| Margins | up | 0.6709 | 0.6718 | yes |
| Margins | flat | 0.3169 | 0.3173 |  |
| Margins | down | 0.0108 | 0.0109 |  |
| Guidance | raised | 0.4803 | 0.4818 | yes |
| Guidance | held | 0.3301 | 0.3311 |  |
| Guidance | cut | 0.0650 | 0.0652 |  |
| Guidance | none | 0.1214 | 0.1218 |  |
| Candor | direct | 0.5286 | 0.5312 | yes |
| Candor | evasive | 0.4665 | 0.4688 |  |
| Next EPS | up | 0.2003 | 0.2018 |  |
| Next EPS | down | 0.7923 | 0.7982 | yes |
| Action | BUY | 0.0664 | 0.0675 |  |
| Action | AVOID | 0.9162 | 0.9325 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.2003 ÷ (0.2003 + 0.7923) = **0.2018**
- **p(BUY)** = 0.0664 ÷ (0.0664 + 0.9162) = **0.0675**. This is the stock score.

The published file has p(BUY) = 0.0675 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `up` 0.4775, `down` 0.2556, `flat` 0.1991, `mixed` 0.0570, `raised` 0.0013 |
| Pricing | `flat` 0.9410, `up` 0.0322, `down` 0.0251, `none` 6.68e-04, `unknown` 3.58e-04 |
| Margins | `up` 0.6709, `flat` 0.3169, `down` 0.0108, `held` 8.90e-04, `stable` 1.13e-04 |
| Guidance | `raised` 0.4803, `held` 0.3301, `none` 0.1214, `cut` 0.0650, `None` 2.20e-04 |
| Candor | `direct` 0.5286, `ev` 0.4665, `mixed` 0.0031, `mix` 1.77e-04, `evapor` 1.22e-04 |
| Next EPS | `down` 0.7923, `up` 0.2003, `unknown` 0.0032, `flat` 0.0029, `none` 6.38e-04 |
| Action | `A` 0.9162, `BUY` 0.0664, `AV` 0.0115, `avoid` 0.0033, `HOLD` 7.37e-04 |
