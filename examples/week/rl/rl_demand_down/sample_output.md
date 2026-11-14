# Sample output: rl_demand_down

Forced-Demand test: RL model with Demand forced to down

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
| Demand | up | 1.48e-05 | 0.0554 |  |
| Demand | flat | 3.16e-05 | 0.1177 |  |
| Demand | down | 2.22e-04 | 0.8269 | forced |
| Pricing | up | 1.12e-05 | 0.0070 |  |
| Pricing | flat | 1.71e-05 | 0.0107 |  |
| Pricing | down | 0.0016 | 0.9823 | yes |
| Margins | up | 2.05e-05 | 0.0737 |  |
| Margins | flat | 3.11e-05 | 0.1119 |  |
| Margins | down | 2.26e-04 | 0.8144 | yes |
| Guidance | raised | 3.47e-06 | 0.0124 |  |
| Guidance | held | 2.30e-05 | 0.0824 |  |
| Guidance | cut | 2.11e-04 | 0.7581 | yes |
| Guidance | none | 4.10e-05 | 0.1470 |  |
| Candor | direct | 8.82e-05 | 0.9395 | yes |
| Candor | evasive | 5.68e-06 | 0.0605 |  |
| Next EPS | up | 2.36e-05 | 0.1461 |  |
| Next EPS | down | 1.38e-04 | 0.8539 | yes |
| Action | BUY | 6.03e-06 | 0.2306 |  |
| Action | AVOID | 2.01e-05 | 0.7694 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 2.36e-05 ÷ (2.36e-05 + 1.38e-04) = **0.1461**
- **p(BUY)** = 6.03e-06 ÷ (6.03e-06 + 2.01e-05) = **0.2306**. This is the stock score.

The published file has p(BUY) = 0.2306 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `<|im_start|>` 7.62e-04, `down` 2.22e-04, `�` 9.76e-05, `cut` 9.24e-05, `usr` 9.03e-05 |
| Pricing | `down` 0.0016, `<|im_start|>` 6.15e-04, `cut` 4.04e-04, `down` 2.45e-04, `-down` 1.76e-04 |
| Margins | `<|im_start|>` 8.01e-04, `down` 2.26e-04, `�` 9.65e-05, `usr` 9.06e-05, `�` 8.45e-05 |
| Guidance | `<|im_start|>` 8.23e-04, `cut` 2.11e-04, `down` 1.87e-04, `�` 9.38e-05, `usr` 8.61e-05 |
| Candor | `<|im_start|>` 7.92e-04, `�` 1.02e-04, `�` 8.96e-05, `direct` 8.82e-05, `�` 8.61e-05 |
| Next EPS | `<|im_start|>` 8.46e-04, `down` 1.38e-04, `�` 9.42e-05, `usr` 8.99e-05, `�` 8.71e-05 |
| Action | `<|im_start|>` 7.99e-04, `down` 1.19e-04, `�` 9.55e-05, `usr` 8.90e-05, `�` 8.49e-05 |
