# Sample output: raw_test (week stage)

Final test of the Call only setup.

Call 218058. The model read the prompt in [examples/prompts/218058_noblock.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/218058_noblock.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: flat
Pricing: flat
Margins: up
Guidance: none
Candor: evasive
Next EPS: up
Action: AVOID
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 0.2624 | 0.2658 |  |
| Demand | flat | 0.7133 | 0.7225 | yes |
| Demand | down | 0.0115 | 0.0117 |  |
| Pricing | up | 0.2913 | 0.2916 |  |
| Pricing | flat | 0.6989 | 0.6996 | yes |
| Pricing | down | 0.0088 | 0.0088 |  |
| Margins | up | 0.6346 | 0.6353 | yes |
| Margins | flat | 0.3397 | 0.3401 |  |
| Margins | down | 0.0246 | 0.0246 |  |
| Guidance | raised | 0.0871 | 0.0877 |  |
| Guidance | held | 0.3446 | 0.3470 |  |
| Guidance | cut | 0.0599 | 0.0603 |  |
| Guidance | none | 0.5014 | 0.5049 | yes |
| Candor | direct | 0.3202 | 0.3208 |  |
| Candor | evasive | 0.6778 | 0.6792 | yes |
| Next EPS | up | 0.6120 | 0.6225 | yes |
| Next EPS | down | 0.3712 | 0.3775 |  |
| Action | BUY | 0.1955 | 0.2018 |  |
| Action | AVOID | 0.7733 | 0.7982 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.6120 ÷ (0.6120 + 0.3712) = **0.6225**
- **p(BUY)** = 0.1955 ÷ (0.1955 + 0.7733) = **0.2018**. This is the stock score.

The published file has p(BUY) = 0.2018 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `flat` 0.7133, `up` 0.2624, `down` 0.0115, `stable` 0.0090, `Flat` 0.0016 |
| Pricing | `flat` 0.6989, `up` 0.2913, `down` 0.0088, `stable` 1.61e-04, `unknown` 1.11e-04 |
| Margins | `up` 0.6346, `flat` 0.3397, `down` 0.0246, `raised` 1.21e-04, `held` 1.14e-04 |
| Guidance | `none` 0.5014, `held` 0.3446, `raised` 0.0871, `cut` 0.0599, `some` 7.54e-04 |
| Candor | `ev` 0.6778, `direct` 0.3202, `mixed` 4.52e-04, `e` 1.47e-04, `indirect` 1.38e-04 |
| Next EPS | `up` 0.6120, `down` 0.3712, `flat` 0.0087, `none` 0.0036, `unknown` 0.0032 |
| Action | `A` 0.7733, `BUY` 0.1955, `AV` 0.0125, `avoid` 0.0086, `HOLD` 0.0046 |
