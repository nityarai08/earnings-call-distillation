# Sample output: rl_shuffled

Control: RL with outcomes shuffled within each month

Call 200826. The model read the prompt in [examples/prompts/200826_rag.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/200826_rag.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: down
Pricing: down
Margins: down
Guidance: cut
Candor: direct
Next EPS: down
Action: AVOID
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 1.87e-05 | 0.0826 |  |
| Demand | flat | 2.79e-05 | 0.1232 |  |
| Demand | down | 1.80e-04 | 0.7942 | yes |
| Pricing | up | 1.97e-05 | 0.0205 |  |
| Pricing | flat | 1.94e-05 | 0.0202 |  |
| Pricing | down | 9.22e-04 | 0.9593 | yes |
| Margins | up | 2.66e-05 | 0.1335 |  |
| Margins | flat | 2.95e-05 | 0.1477 |  |
| Margins | down | 1.43e-04 | 0.7188 | yes |
| Guidance | raised | 4.50e-06 | 0.0210 |  |
| Guidance | held | 4.27e-05 | 0.1990 |  |
| Guidance | cut | 1.39e-04 | 0.6474 | yes |
| Guidance | none | 2.85e-05 | 0.1326 |  |
| Candor | direct | 6.48e-05 | 0.9322 | yes |
| Candor | evasive | 4.72e-06 | 0.0678 |  |
| Next EPS | up | 2.58e-05 | 0.1901 |  |
| Next EPS | down | 1.10e-04 | 0.8099 | yes |
| Action | BUY | 4.12e-06 | 0.1321 |  |
| Action | AVOID | 2.71e-05 | 0.8679 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 2.58e-05 ÷ (2.58e-05 + 1.10e-04) = **0.1901**
- **p(BUY)** = 4.12e-06 ÷ (4.12e-06 + 2.71e-05) = **0.1321**. This is the stock score.

The published file has p(BUY) = 0.1321 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `<|im_start|>` 8.31e-04, `down` 1.80e-04, `cut` 1.09e-04, `usr` 9.11e-05, `�` 8.76e-05 |
| Pricing | `down` 9.22e-04, `<|im_start|>` 6.54e-04, `cut` 4.29e-04, `down` 1.65e-04, `cuts` 1.23e-04 |
| Margins | `<|im_start|>` 8.78e-04, `down` 1.43e-04, `usr` 9.25e-05, `�` 8.90e-05, `�` 7.79e-05 |
| Guidance | `<|im_start|>` 8.65e-04, `cut` 1.39e-04, `down` 1.03e-04, `usr` 8.90e-05, `�` 8.56e-05 |
| Candor | `<|im_start|>` 9.02e-04, `�` 9.97e-05, `usr` 8.86e-05, `�` 8.59e-05, `�` 8.52e-05 |
| Next EPS | `<|im_start|>` 8.77e-04, `down` 1.10e-04, `usr` 9.18e-05, `�` 9.03e-05, `�` 8.16e-05 |
| Action | `<|im_start|>` 8.49e-04, `cut` 1.45e-04, `down` 1.13e-04, `usr` 9.09e-05, `�` 8.34e-05 |
