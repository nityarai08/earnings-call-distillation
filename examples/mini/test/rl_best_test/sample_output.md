# Sample output: rl_best_test (mini stage)

Final test of the RL setup.

Call 216962. The model read the prompt in [examples/prompts/216962_rag.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/216962_rag.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: up
Pricing: up
Margins: up
Guidance: none
Candor: direct
Next EPS: down
Action: BUY
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 6.63e-04 | 0.9582 | yes |
| Demand | flat | 2.27e-05 | 0.0328 |  |
| Demand | down | 6.21e-06 | 0.0090 |  |
| Pricing | up | 1.19e-04 | 0.6182 | yes |
| Pricing | flat | 5.38e-05 | 0.2786 |  |
| Pricing | down | 1.99e-05 | 0.1032 |  |
| Margins | up | 9.33e-05 | 0.4917 | yes |
| Margins | flat | 6.72e-05 | 0.3542 |  |
| Margins | down | 2.92e-05 | 0.1541 |  |
| Guidance | raised | 4.54e-06 | 0.0315 |  |
| Guidance | held | 2.29e-05 | 0.1587 |  |
| Guidance | cut | 4.11e-05 | 0.2852 |  |
| Guidance | none | 7.55e-05 | 0.5246 | yes |
| Candor | direct | 1.38e-04 | 0.9767 | yes |
| Candor | evasive | 3.29e-06 | 0.0233 |  |
| Next EPS | up | 6.75e-05 | 0.4244 |  |
| Next EPS | down | 9.15e-05 | 0.5756 | yes |
| Action | BUY | 1.86e-05 | 0.6254 | yes |
| Action | AVOID | 1.11e-05 | 0.3746 |  |

## The two numbers that the scores use

- **p(Next EPS up)** = 6.75e-05 ÷ (6.75e-05 + 9.15e-05) = **0.4244**
- **p(BUY)** = 1.86e-05 ÷ (1.86e-05 + 1.11e-05) = **0.6254**. This is the stock score.

The published file has p(BUY) = 0.6254 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `<|im_start|>` 8.79e-04, `up` 6.63e-04, `' '` 5.95e-04, `up` 4.28e-04, `,` 4.15e-04 |
| Pricing | `<|im_start|>` 8.96e-04, `' '` 2.16e-04, `-` 2.06e-04, `�` 1.97e-04, `1` 1.91e-04 |
| Margins | `<|im_start|>` 8.19e-04, `�` 1.94e-04, `�` 1.91e-04, `-` 1.88e-04, `' '` 1.74e-04 |
| Guidance | `<|im_start|>` 7.39e-04, `�` 2.29e-04, `�` 2.02e-04, `�` 1.84e-04, `�` 1.78e-04 |
| Candor | `<|im_start|>` 6.28e-04, `�` 2.42e-04, `�` 2.04e-04, `�` 1.98e-04, `�` 1.91e-04 |
| Next EPS | `<|im_start|>` 6.26e-04, `�` 2.30e-04, `�` 1.97e-04, `�` 1.94e-04, `-` 1.88e-04 |
| Action | `<|im_start|>` 6.70e-04, `�` 2.04e-04, `�` 1.83e-04, `�` 1.83e-04, `�` 1.80e-04 |
