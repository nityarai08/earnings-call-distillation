# Sample output: sft_best_check

SFT setup on the check calls

Call 62273. The model read the prompt in [examples/prompts/62273_rag_noreaction.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/62273_rag_noreaction.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: down
Pricing: flat
Margins: down
Guidance: held
Candor: evasive
Next EPS: down
Action: AVOID
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 4.54e-05 | 0.3182 |  |
| Demand | flat | 4.16e-05 | 0.2920 |  |
| Demand | down | 5.56e-05 | 0.3898 | yes |
| Pricing | up | 2.14e-05 | 0.0855 |  |
| Pricing | flat | 1.51e-04 | 0.6053 | yes |
| Pricing | down | 7.73e-05 | 0.3092 |  |
| Margins | up | 3.74e-05 | 0.2060 |  |
| Margins | flat | 7.16e-05 | 0.3939 |  |
| Margins | down | 7.27e-05 | 0.4001 | yes |
| Guidance | raised | 4.50e-06 | 0.0325 |  |
| Guidance | held | 6.03e-05 | 0.4350 | yes |
| Guidance | cut | 4.69e-05 | 0.3388 |  |
| Guidance | none | 2.69e-05 | 0.1938 |  |
| Candor | direct | 1.80e-05 | 0.4858 |  |
| Candor | evasive | 1.90e-05 | 0.5142 | yes |
| Next EPS | up | 2.65e-05 | 0.1656 |  |
| Next EPS | down | 1.34e-04 | 0.8344 | yes |
| Action | BUY | 6.02e-06 | 0.2341 |  |
| Action | AVOID | 1.97e-05 | 0.7659 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 2.65e-05 ÷ (2.65e-05 + 1.34e-04) = **0.1656**
- **p(BUY)** = 6.02e-06 ÷ (6.02e-06 + 1.97e-05) = **0.2341**. This is the stock score.

The published file has p(BUY) = 0.2341 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `<|im_start|>` 9.70e-04, `�` 9.75e-05, `usr` 8.95e-05, `�` 8.21e-05, `�` 7.96e-05 |
| Pricing | `<|im_start|>` 8.57e-04, `flat` 1.51e-04, `�` 1.02e-04, `�` 8.76e-05, `�` 8.69e-05 |
| Margins | `<|im_start|>` 9.00e-04, `�` 9.63e-05, `usr` 8.84e-05, `�` 8.05e-05, `�` 7.92e-05 |
| Guidance | `<|im_start|>` 9.07e-04, `down` 1.03e-04, `�` 9.12e-05, `usr` 8.84e-05, `�` 8.05e-05 |
| Candor | `<|im_start|>` 0.0011, `usr` 9.24e-05, `�` 8.28e-05, `�` 8.09e-05, `�` 7.66e-05 |
| Next EPS | `<|im_start|>` 9.43e-04, `down` 1.34e-04, `usr` 9.34e-05, `�` 9.19e-05, `�` 7.80e-05 |
| Action | `<|im_start|>` 9.58e-04, `down` 1.04e-04, `usr` 9.06e-05, `�` 8.98e-05, `�` 7.75e-05 |
