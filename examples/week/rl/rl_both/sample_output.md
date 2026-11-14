# Sample output: rl_both

RL with the business and trade rewards together

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
| Demand | up | 1.10e-05 | 0.0431 |  |
| Demand | flat | 2.63e-05 | 0.1032 |  |
| Demand | down | 2.18e-04 | 0.8538 | yes |
| Pricing | up | 6.80e-06 | 0.0041 |  |
| Pricing | flat | 1.33e-05 | 0.0080 |  |
| Pricing | down | 0.0017 | 0.9880 | yes |
| Margins | up | 1.92e-05 | 0.1022 |  |
| Margins | flat | 2.73e-05 | 0.1453 |  |
| Margins | down | 1.42e-04 | 0.7525 | yes |
| Guidance | raised | 3.03e-06 | 0.0135 |  |
| Guidance | held | 1.91e-05 | 0.0849 |  |
| Guidance | cut | 1.76e-04 | 0.7825 | yes |
| Guidance | none | 2.68e-05 | 0.1191 |  |
| Candor | direct | 4.99e-05 | 0.8741 | yes |
| Candor | evasive | 7.19e-06 | 0.1259 |  |
| Next EPS | up | 1.85e-05 | 0.1266 |  |
| Next EPS | down | 1.28e-04 | 0.8734 | yes |
| Action | BUY | 5.66e-06 | 0.2672 |  |
| Action | AVOID | 1.55e-05 | 0.7328 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 1.85e-05 ÷ (1.85e-05 + 1.28e-04) = **0.1266**
- **p(BUY)** = 5.66e-06 ÷ (5.66e-06 + 1.55e-05) = **0.2672**. This is the stock score.

The published file has p(BUY) = 0.2672 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `<|im_start|>` 8.62e-04, `down` 2.18e-04, `cut` 1.20e-04, `usr` 9.09e-05, `�` 9.02e-05 |
| Pricing | `down` 0.0017, `<|im_start|>` 6.47e-04, `cut` 4.96e-04, `down` 2.78e-04, `-down` 1.77e-04 |
| Margins | `<|im_start|>` 9.83e-04, `down` 1.42e-04, `usr` 9.21e-05, `�` 9.14e-05, `�` 8.13e-05 |
| Guidance | `<|im_start|>` 9.51e-04, `cut` 1.76e-04, `down` 1.63e-04, `usr` 8.85e-05, `�` 8.78e-05 |
| Candor | `<|im_start|>` 0.0010, `�` 9.40e-05, `usr` 8.83e-05, `�` 8.49e-05, `�` 8.30e-05 |
| Next EPS | `<|im_start|>` 0.0010, `down` 1.28e-04, `usr` 9.29e-05, `�` 8.93e-05, `�` 8.32e-05 |
| Action | `<|im_start|>` 9.56e-04, `down` 1.40e-04, `cut` 9.39e-05, `�` 9.17e-05, `usr` 8.96e-05 |
