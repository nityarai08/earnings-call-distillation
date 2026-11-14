# Sample output: rl_business

RL with the business reward (EPS forecast)

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
| Demand | up | 1.21e-05 | 0.0504 |  |
| Demand | flat | 3.29e-05 | 0.1373 |  |
| Demand | down | 1.94e-04 | 0.8122 | yes |
| Pricing | up | 9.69e-06 | 0.0092 |  |
| Pricing | flat | 1.65e-05 | 0.0156 |  |
| Pricing | down | 0.0010 | 0.9752 | yes |
| Margins | up | 1.81e-05 | 0.0848 |  |
| Margins | flat | 2.79e-05 | 0.1306 |  |
| Margins | down | 1.67e-04 | 0.7846 | yes |
| Guidance | raised | 3.53e-06 | 0.0128 |  |
| Guidance | held | 2.11e-05 | 0.0768 |  |
| Guidance | cut | 2.13e-04 | 0.7756 | yes |
| Guidance | none | 3.71e-05 | 0.1348 |  |
| Candor | direct | 4.97e-05 | 0.8594 | yes |
| Candor | evasive | 8.13e-06 | 0.1406 |  |
| Next EPS | up | 1.94e-05 | 0.1373 |  |
| Next EPS | down | 1.22e-04 | 0.8627 | yes |
| Action | BUY | 5.59e-06 | 0.2369 |  |
| Action | AVOID | 1.80e-05 | 0.7631 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 1.94e-05 ÷ (1.94e-05 + 1.22e-04) = **0.1373**
- **p(BUY)** = 5.59e-06 ÷ (5.59e-06 + 1.80e-05) = **0.2369**. This is the stock score.

The published file has p(BUY) = 0.2369 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `<|im_start|>` 9.27e-04, `down` 1.94e-04, `cut` 1.02e-04, `�` 9.11e-05, `usr` 8.90e-05 |
| Pricing | `down` 0.0010, `<|im_start|>` 7.52e-04, `cut` 4.78e-04, `down` 1.76e-04, `cuts` 1.41e-04 |
| Margins | `<|im_start|>` 0.0010, `down` 1.67e-04, `�` 9.03e-05, `usr` 8.82e-05, `�` 8.03e-05 |
| Guidance | `<|im_start|>` 9.87e-04, `cut` 2.13e-04, `down` 1.61e-04, `�` 8.62e-05, `usr` 8.49e-05 |
| Candor | `<|im_start|>` 9.16e-04, `�` 9.80e-05, `�` 8.72e-05, `�` 8.52e-05, `usr` 8.45e-05 |
| Next EPS | `<|im_start|>` 0.0010, `down` 1.22e-04, `usr` 8.92e-05, `�` 8.85e-05, `�` 8.44e-05 |
| Action | `<|im_start|>` 9.62e-04, `down` 1.48e-04, `cut` 1.19e-04, `�` 8.81e-05, `usr` 8.54e-05 |
