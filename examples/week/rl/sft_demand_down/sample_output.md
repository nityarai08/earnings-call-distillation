# Sample output: sft_demand_down

Forced-Demand test: SFT model with Demand forced to down

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
| Demand | up | 1.87e-05 | 0.0964 |  |
| Demand | flat | 4.03e-05 | 0.2074 |  |
| Demand | down | 1.35e-04 | 0.6962 | forced |
| Pricing | up | 1.30e-05 | 0.0228 |  |
| Pricing | flat | 2.26e-05 | 0.0395 |  |
| Pricing | down | 5.37e-04 | 0.9378 | yes |
| Margins | up | 3.34e-05 | 0.1790 |  |
| Margins | flat | 3.76e-05 | 0.2012 |  |
| Margins | down | 1.16e-04 | 0.6198 | yes |
| Guidance | raised | 4.21e-06 | 0.0244 |  |
| Guidance | held | 2.74e-05 | 0.1588 |  |
| Guidance | cut | 9.29e-05 | 0.5392 | yes |
| Guidance | none | 4.78e-05 | 0.2776 |  |
| Candor | direct | 3.29e-05 | 0.7783 | yes |
| Candor | evasive | 9.38e-06 | 0.2217 |  |
| Next EPS | up | 2.91e-05 | 0.1950 |  |
| Next EPS | down | 1.20e-04 | 0.8050 | yes |
| Action | BUY | 5.58e-06 | 0.2214 |  |
| Action | AVOID | 1.96e-05 | 0.7786 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 2.91e-05 ÷ (2.91e-05 + 1.20e-04) = **0.1950**
- **p(BUY)** = 5.58e-06 ÷ (5.58e-06 + 1.96e-05) = **0.2214**. This is the stock score.

The published file has p(BUY) = 0.2213 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `<|im_start|>` 9.24e-04, `down` 1.35e-04, `�` 9.73e-05, `usr` 9.36e-05, `�` 8.20e-05 |
| Pricing | `<|im_start|>` 8.18e-04, `down` 5.37e-04, `cut` 1.80e-04, `down` 1.09e-04, `�` 9.77e-05 |
| Margins | `<|im_start|>` 9.39e-04, `down` 1.16e-04, `�` 9.59e-05, `usr` 9.23e-05, `�` 7.83e-05 |
| Guidance | `<|im_start|>` 9.68e-04, `down` 1.25e-04, `cut` 9.29e-05, `�` 9.15e-05, `usr` 9.01e-05 |
| Candor | `<|im_start|>` 9.93e-04, `�` 9.24e-05, `usr` 8.95e-05, `�` 8.28e-05, `�` 8.03e-05 |
| Next EPS | `<|im_start|>` 9.46e-04, `down` 1.20e-04, `�` 9.51e-05, `usr` 9.22e-05, `�` 8.20e-05 |
| Action | `<|im_start|>` 9.46e-04, `down` 1.22e-04, `usr` 9.22e-05, `�` 9.15e-05, `�` 7.83e-05 |
