# Sample output: sft_demand_up

Forced-Demand test: SFT model with Demand forced to up

Call 200826. The model read the prompt in [examples/prompts/200826_rag.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/200826_rag.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: up
Pricing: down
Margins: up
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
| Demand | up | 1.87e-05 | 0.0964 | forced |
| Demand | flat | 4.03e-05 | 0.2074 |  |
| Demand | down | 1.35e-04 | 0.6962 |  |
| Pricing | up | 2.49e-05 | 0.1085 |  |
| Pricing | flat | 3.56e-05 | 0.1555 |  |
| Pricing | down | 1.69e-04 | 0.7360 | yes |
| Margins | up | 7.32e-05 | 0.4556 | yes |
| Margins | flat | 3.59e-05 | 0.2238 |  |
| Margins | down | 5.15e-05 | 0.3206 |  |
| Guidance | raised | 6.30e-06 | 0.0398 |  |
| Guidance | held | 5.44e-05 | 0.3435 |  |
| Guidance | cut | 2.33e-05 | 0.1472 |  |
| Guidance | none | 7.44e-05 | 0.4695 | yes |
| Candor | direct | 3.89e-05 | 0.8416 | yes |
| Candor | evasive | 7.31e-06 | 0.1584 |  |
| Next EPS | up | 4.10e-05 | 0.3558 |  |
| Next EPS | down | 7.42e-05 | 0.6442 | yes |
| Action | BUY | 6.74e-06 | 0.2689 |  |
| Action | AVOID | 1.83e-05 | 0.7311 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 4.10e-05 ÷ (4.10e-05 + 7.42e-05) = **0.3558**
- **p(BUY)** = 6.74e-06 ÷ (6.74e-06 + 1.83e-05) = **0.2689**. This is the stock score.

The published file has p(BUY) = 0.2689 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `<|im_start|>` 9.24e-04, `down` 1.35e-04, `�` 9.73e-05, `usr` 9.36e-05, `�` 8.20e-05 |
| Pricing | `<|im_start|>` 8.84e-04, `down` 1.69e-04, `�` 9.61e-05, `usr` 9.31e-05, `�` 8.09e-05 |
| Margins | `<|im_start|>` 9.19e-04, `�` 9.69e-05, `usr` 9.25e-05, `�` 7.85e-05, `�` 7.61e-05 |
| Guidance | `<|im_start|>` 9.64e-04, `�` 9.70e-05, `usr` 9.11e-05, `�` 8.30e-05, `�` 7.79e-05 |
| Candor | `<|im_start|>` 9.79e-04, `�` 9.54e-05, `usr` 8.76e-05, `�` 8.49e-05, `�` 8.10e-05 |
| Next EPS | `<|im_start|>` 9.54e-04, `�` 9.75e-05, `usr` 9.09e-05, `�` 8.14e-05, `�` 8.08e-05 |
| Action | `<|im_start|>` 9.81e-04, `�` 9.34e-05, `usr` 9.34e-05, `down` 8.91e-05, `�` 7.99e-05 |
