# Sample output: rl_best_check

RL setup on the check calls

Call 62273. The model read the prompt in [examples/prompts/62273_rag_noreaction.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/62273_rag_noreaction.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: up
Pricing: flat
Margins: flat
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
| Demand | up | 6.14e-05 | 0.3976 | yes |
| Demand | flat | 4.09e-05 | 0.2649 |  |
| Demand | down | 5.21e-05 | 0.3375 |  |
| Pricing | up | 2.73e-05 | 0.0706 |  |
| Pricing | flat | 3.34e-04 | 0.8636 | yes |
| Pricing | down | 2.55e-05 | 0.0658 |  |
| Margins | up | 7.21e-05 | 0.3963 |  |
| Margins | flat | 8.17e-05 | 0.4491 | yes |
| Margins | down | 2.81e-05 | 0.1546 |  |
| Guidance | raised | 4.98e-06 | 0.0109 |  |
| Guidance | held | 4.28e-04 | 0.9368 | yes |
| Guidance | cut | 9.65e-06 | 0.0211 |  |
| Guidance | none | 1.42e-05 | 0.0311 |  |
| Candor | direct | 2.68e-05 | 0.6875 | yes |
| Candor | evasive | 1.22e-05 | 0.3125 |  |
| Next EPS | up | 3.01e-05 | 0.2372 |  |
| Next EPS | down | 9.67e-05 | 0.7628 | yes |
| Action | BUY | 7.40e-06 | 0.2659 |  |
| Action | AVOID | 2.04e-05 | 0.7341 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 3.01e-05 ÷ (3.01e-05 + 9.67e-05) = **0.2372**
- **p(BUY)** = 7.40e-06 ÷ (7.40e-06 + 2.04e-05) = **0.2659**. This is the stock score.

The published file has p(BUY) = 0.2659 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `<|im_start|>` 8.08e-04, `�` 9.66e-05, `usr` 8.86e-05, `�` 8.26e-05, `�` 8.19e-05 |
| Pricing | `<|im_start|>` 6.05e-04, `flat` 3.34e-04, `held` 1.53e-04, `flat` 1.14e-04, `�` 1.12e-04 |
| Margins | `<|im_start|>` 7.06e-04, `�` 1.03e-04, `held` 8.76e-05, `�` 8.70e-05, `�` 8.63e-05 |
| Guidance | `<|im_start|>` 5.94e-04, `held` 4.28e-04, `hold` 1.70e-04, `flat` 1.10e-04, `holds` 1.02e-04 |
| Candor | `<|im_start|>` 9.09e-04, `�` 8.93e-05, `usr` 8.86e-05, `�` 8.66e-05, `�` 8.59e-05 |
| Next EPS | `<|im_start|>` 8.35e-04, `down` 9.67e-05, `�` 9.23e-05, `usr` 9.09e-05, `�` 8.34e-05 |
| Action | `<|im_start|>` 8.26e-04, `�` 9.71e-05, `usr` 8.91e-05, `�` 8.78e-05, `�` 8.31e-05 |
