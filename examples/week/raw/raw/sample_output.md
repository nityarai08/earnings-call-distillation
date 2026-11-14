# Sample output: raw

Call only setup: the call text, no knowledge block

Call 200826. The model read the prompt in [examples/prompts/200826_noblock.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/200826_noblock.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: down
Pricing: down
Margins: up
Guidance: held
Candor: direct
Next EPS: down
Action: AVOID
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 0.2198 | 0.2214 |  |
| Demand | flat | 0.3624 | 0.3650 |  |
| Demand | down | 0.4106 | 0.4136 | yes |
| Pricing | up | 0.0392 | 0.0394 |  |
| Pricing | flat | 0.0646 | 0.0649 |  |
| Pricing | down | 0.8921 | 0.8958 | yes |
| Margins | up | 0.5975 | 0.5991 | yes |
| Margins | flat | 0.2822 | 0.2830 |  |
| Margins | down | 0.1176 | 0.1180 |  |
| Guidance | raised | 0.2449 | 0.2461 |  |
| Guidance | held | 0.3145 | 0.3160 | yes |
| Guidance | cut | 0.2449 | 0.2461 |  |
| Guidance | none | 0.1907 | 0.1917 |  |
| Candor | direct | 0.7768 | 0.7773 | yes |
| Candor | evasive | 0.2226 | 0.2227 |  |
| Next EPS | up | 0.0671 | 0.0675 |  |
| Next EPS | down | 0.9260 | 0.9325 | yes |
| Action | BUY | 0.0051 | 0.0052 |  |
| Action | AVOID | 0.9697 | 0.9948 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.0671 ÷ (0.0671 + 0.9260) = **0.0675**
- **p(BUY)** = 0.0051 ÷ (0.0051 + 0.9697) = **0.0052**. This is the stock score.

The published file has p(BUY) = 0.0052 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `down` 0.4106, `flat` 0.3624, `up` 0.2198, `cut` 0.0017, `raised` 0.0011 |
| Pricing | `down` 0.8921, `flat` 0.0646, `up` 0.0392, `mixed` 0.0032, `cut` 1.93e-04 |
| Margins | `up` 0.5975, `flat` 0.2822, `down` 0.1176, `raised` 4.24e-04, `held` 3.74e-04 |
| Guidance | `held` 0.3145, `cut` 0.2449, `raised` 0.2449, `none` 0.1907, `flat` 3.46e-04 |
| Candor | `direct` 0.7768, `ev` 0.2226, `indirect` 5.81e-05, `e` 5.46e-05, `evapor` 3.75e-05 |
| Next EPS | `down` 0.9260, `up` 0.0671, `flat` 0.0033, `unknown` 0.0026, `none` 3.52e-04 |
| Action | `A` 0.9697, `avoid` 0.0138, `AV` 0.0095, `BUY` 0.0051, `down` 8.84e-04 |
