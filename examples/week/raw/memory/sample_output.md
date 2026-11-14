# Sample output: memory

No call setup: the facts line, no call text

Call 200826. The model read the prompt in [examples/prompts/200826_nocall_noblock.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/200826_nocall_noblock.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: down
Pricing: flat
Margins: down
Guidance: none
Candor: evasive
Next EPS: down
Action: AVOID
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 0.0300 | 0.0301 |  |
| Demand | flat | 0.2849 | 0.2854 |  |
| Demand | down | 0.6835 | 0.6846 | yes |
| Pricing | up | 0.0951 | 0.0952 |  |
| Pricing | flat | 0.7965 | 0.7970 | yes |
| Pricing | down | 0.1078 | 0.1079 |  |
| Margins | up | 0.0254 | 0.0254 |  |
| Margins | flat | 0.3968 | 0.3970 |  |
| Margins | down | 0.5774 | 0.5776 | yes |
| Guidance | raised | 4.95e-05 | 0.0000 |  |
| Guidance | held | 0.0073 | 0.0074 |  |
| Guidance | cut | 0.0290 | 0.0291 |  |
| Guidance | none | 0.9614 | 0.9635 | yes |
| Candor | direct | 0.2346 | 0.2942 |  |
| Candor | evasive | 0.5628 | 0.7058 | yes |
| Next EPS | up | 0.0046 | 0.0046 |  |
| Next EPS | down | 0.9940 | 0.9954 | yes |
| Action | BUY | 2.75e-05 | 0.0000 |  |
| Action | AVOID | 0.9974 | 1.0000 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.0046 ÷ (0.0046 + 0.9940) = **0.0046**
- **p(BUY)** = 2.75e-05 ÷ (2.75e-05 + 0.9974) = **0.0000**. This is the stock score.

The published file has p(BUY) = 0.0000 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `down` 0.6835, `flat` 0.2849, `up` 0.0300, `Down` 4.56e-04, `Flat` 1.23e-04 |
| Pricing | `flat` 0.7965, `down` 0.1078, `up` 0.0951, `stable` 5.26e-05, `neutral` 4.10e-05 |
| Margins | `down` 0.5774, `flat` 0.3968, `up` 0.0254, `stable` 6.29e-05, `cut` 3.58e-05 |
| Guidance | `none` 0.9614, `cut` 0.0290, `held` 0.0073, `no` 4.14e-04, `None` 3.23e-04 |
| Candor | `ev` 0.5628, `direct` 0.2346, `none` 0.1612, `not` 0.0218, `unknown` 0.0091 |
| Next EPS | `down` 0.9940, `up` 0.0046, `flat` 0.0012, `unknown` 1.23e-04, `none` 3.51e-05 |
| Action | `A` 0.9974, `avoid` 0.0017, `AV` 8.03e-04, `Avoid` 7.47e-05, `BUY` 2.75e-05 |
