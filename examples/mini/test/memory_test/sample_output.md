# Sample output: memory_test (mini stage)

Final test of the Memory only setup.

Call 216962. The model read the prompt in [examples/prompts/216962_nocall_noblock.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/216962_nocall_noblock.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: flat
Pricing: flat
Margins: flat
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
| Demand | up | 0.1120 | 0.1122 |  |
| Demand | flat | 0.4430 | 0.4439 | yes |
| Demand | down | 0.4430 | 0.4439 |  |
| Pricing | up | 0.0642 | 0.0642 |  |
| Pricing | flat | 0.7816 | 0.7819 | yes |
| Pricing | down | 0.1539 | 0.1540 |  |
| Margins | up | 0.0228 | 0.0228 |  |
| Margins | flat | 0.5189 | 0.5191 | yes |
| Margins | down | 0.4579 | 0.4581 |  |
| Guidance | raised | 5.77e-05 | 0.0001 |  |
| Guidance | held | 0.0059 | 0.0059 |  |
| Guidance | cut | 0.0025 | 0.0025 |  |
| Guidance | none | 0.9904 | 0.9916 | yes |
| Candor | direct | 0.2881 | 0.3208 |  |
| Candor | evasive | 0.6099 | 0.6792 | yes |
| Next EPS | up | 0.0097 | 0.0097 |  |
| Next EPS | down | 0.9853 | 0.9903 | yes |
| Action | BUY | 7.41e-05 | 0.0001 |  |
| Action | AVOID | 0.9896 | 0.9999 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.0097 ÷ (0.0097 + 0.9853) = **0.0097**
- **p(BUY)** = 7.41e-05 ÷ (7.41e-05 + 0.9896) = **0.0001**. This is the stock score.

The published file has p(BUY) = 0.0001 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `down` 0.4430, `flat` 0.4430, `up` 0.1120, `Down` 4.04e-04, `none` 2.45e-04 |
| Pricing | `flat` 0.7816, `down` 0.1539, `up` 0.0642, `neutral` 2.29e-05, `stable` 2.02e-05 |
| Margins | `flat` 0.5189, `down` 0.4579, `up` 0.0228, `cut` 7.72e-05, `fl` 3.43e-05 |
| Guidance | `none` 0.9904, `held` 0.0059, `cut` 0.0025, `no` 2.75e-04, `None` 1.78e-04 |
| Candor | `ev` 0.6099, `direct` 0.2881, `none` 0.0825, `not` 0.0099, `unknown` 0.0022 |
| Next EPS | `down` 0.9853, `up` 0.0097, `flat` 0.0046, `unknown` 2.00e-04, `none` 5.07e-05 |
| Action | `A` 0.9896, `avoid` 0.0076, `AV` 0.0025, `Avoid` 2.01e-04, `BUY` 7.41e-05 |
