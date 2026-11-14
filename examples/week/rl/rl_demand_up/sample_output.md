# Sample output: rl_demand_up

Forced-Demand test: RL model with Demand forced to up

Call 200826. The model read the prompt in [examples/prompts/200826_rag.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/200826_rag.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: up
Pricing: down
Margins: down
Guidance: none
Candor: direct
Next EPS: down
Action: AVOID
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 1.48e-05 | 0.0554 | forced |
| Demand | flat | 3.16e-05 | 0.1177 |  |
| Demand | down | 2.22e-04 | 0.8269 |  |
| Pricing | up | 2.07e-05 | 0.0614 |  |
| Pricing | flat | 2.94e-05 | 0.0873 |  |
| Pricing | down | 2.87e-04 | 0.8513 | yes |
| Margins | up | 6.07e-05 | 0.3774 |  |
| Margins | flat | 3.65e-05 | 0.2271 |  |
| Margins | down | 6.36e-05 | 0.3955 | yes |
| Guidance | raised | 4.48e-06 | 0.0272 |  |
| Guidance | held | 4.31e-05 | 0.2620 |  |
| Guidance | cut | 5.81e-05 | 0.3526 |  |
| Guidance | none | 5.90e-05 | 0.3582 | yes |
| Candor | direct | 8.79e-05 | 0.9406 | yes |
| Candor | evasive | 5.55e-06 | 0.0594 |  |
| Next EPS | up | 3.18e-05 | 0.2783 |  |
| Next EPS | down | 8.25e-05 | 0.7217 | yes |
| Action | BUY | 6.57e-06 | 0.2462 |  |
| Action | AVOID | 2.01e-05 | 0.7538 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 3.18e-05 ÷ (3.18e-05 + 8.25e-05) = **0.2783**
- **p(BUY)** = 6.57e-06 ÷ (6.57e-06 + 2.01e-05) = **0.2462**. This is the stock score.

The published file has p(BUY) = 0.2462 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `<|im_start|>` 7.62e-04, `down` 2.22e-04, `�` 9.76e-05, `cut` 9.24e-05, `usr` 9.03e-05 |
| Pricing | `<|im_start|>` 7.33e-04, `down` 2.87e-04, `�` 1.01e-04, `�` 9.25e-05, `usr` 8.96e-05 |
| Margins | `<|im_start|>` 8.44e-04, `�` 9.62e-05, `usr` 9.18e-05, `�` 7.92e-05, `�` 7.92e-05 |
| Guidance | `<|im_start|>` 8.53e-04, `�` 9.72e-05, `down` 9.57e-05, `usr` 8.78e-05, `�` 8.38e-05 |
| Candor | `<|im_start|>` 8.15e-04, `�` 1.00e-04, `�` 8.86e-05, `�` 8.79e-05, `direct` 8.79e-05 |
| Next EPS | `<|im_start|>` 8.53e-04, `�` 9.49e-05, `usr` 8.92e-05, `�` 8.58e-05, `�` 8.25e-05 |
| Action | `<|im_start|>` 8.09e-04, `�` 9.66e-05, `down` 9.00e-05, `usr` 9.00e-05, `�` 8.59e-05 |
