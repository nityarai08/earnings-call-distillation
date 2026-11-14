# Sample output: rl_best_test (week stage)

Final test of the RL setup.

Call 218058. The model read the prompt in [examples/prompts/218058_rag.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/218058_rag.txt). The prompt ends with the label `Demand:`.

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
| Demand | up | 6.68e-04 | 0.9180 | yes |
| Demand | flat | 5.40e-05 | 0.0742 |  |
| Demand | down | 5.66e-06 | 0.0078 |  |
| Pricing | up | 8.68e-05 | 0.3617 |  |
| Pricing | flat | 1.41e-04 | 0.5871 | yes |
| Pricing | down | 1.23e-05 | 0.0512 |  |
| Margins | up | 1.72e-04 | 0.7136 | yes |
| Margins | flat | 5.07e-05 | 0.2109 |  |
| Margins | down | 1.82e-05 | 0.0755 |  |
| Guidance | raised | 4.98e-06 | 0.0111 |  |
| Guidance | held | 1.48e-04 | 0.3301 |  |
| Guidance | cut | 5.58e-06 | 0.0125 |  |
| Guidance | none | 2.90e-04 | 0.6463 | yes |
| Candor | direct | 5.72e-05 | 0.8766 | yes |
| Candor | evasive | 8.05e-06 | 0.1234 |  |
| Next EPS | up | 5.22e-05 | 0.6645 | yes |
| Next EPS | down | 2.64e-05 | 0.3355 |  |
| Action | BUY | 3.16e-05 | 0.6715 | yes |
| Action | AVOID | 1.54e-05 | 0.3285 |  |

## The two numbers that the scores use

- **p(Next EPS up)** = 5.22e-05 ÷ (5.22e-05 + 2.64e-05) = **0.6645**
- **p(BUY)** = 3.16e-05 ÷ (3.16e-05 + 1.54e-05) = **0.6715**. This is the stock score.

The published file has p(BUY) = 0.6715 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `up` 6.68e-04, `<|im_start|>` 5.80e-04, `up` 1.71e-04, `-up` 1.47e-04, `�` 1.36e-04 |
| Pricing | `<|im_start|>` 6.31e-04, `flat` 1.41e-04, `�` 1.15e-04, `held` 1.13e-04, `�` 9.45e-05 |
| Margins | `<|im_start|>` 7.22e-04, `up` 1.72e-04, `�` 1.06e-04, `usr` 8.50e-05, `�` 8.36e-05 |
| Guidance | `<|im_start|>` 7.28e-04, `none` 2.90e-04, `held` 1.48e-04, `�` 1.23e-04, `none` 1.19e-04 |
| Candor | `<|im_start|>` 8.54e-04, `�` 9.88e-05, `�` 8.86e-05, `�` 8.65e-05, `usr` 8.45e-05 |
| Next EPS | `<|im_start|>` 8.36e-04, `�` 1.01e-04, `usr` 8.74e-05, `�` 8.61e-05, `�` 8.47e-05 |
| Action | `<|im_start|>` 7.47e-04, `�` 1.11e-04, `�` 9.13e-05, `buy` 8.78e-05, `�` 8.65e-05 |
