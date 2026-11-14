# Sample output: sft_best_test (week stage)

Final test of the SFT setup.

Call 218058. The model read the prompt in [examples/prompts/218058_rag.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/218058_rag.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: up
Pricing: flat
Margins: up
Guidance: none
Candor: direct
Next EPS: up
Action: BUY
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 2.54e-04 | 0.8255 | yes |
| Demand | flat | 4.89e-05 | 0.1588 |  |
| Demand | down | 4.84e-06 | 0.0157 |  |
| Pricing | up | 7.91e-05 | 0.4157 |  |
| Pricing | flat | 9.77e-05 | 0.5133 | yes |
| Pricing | down | 1.35e-05 | 0.0710 |  |
| Margins | up | 1.17e-04 | 0.6592 | yes |
| Margins | flat | 4.18e-05 | 0.2350 |  |
| Margins | down | 1.88e-05 | 0.1057 |  |
| Guidance | raised | 6.93e-06 | 0.0218 |  |
| Guidance | held | 1.04e-04 | 0.3266 |  |
| Guidance | cut | 6.99e-06 | 0.0220 |  |
| Guidance | none | 2.00e-04 | 0.6296 | yes |
| Candor | direct | 3.10e-05 | 0.7448 | yes |
| Candor | evasive | 1.06e-05 | 0.2552 |  |
| Next EPS | up | 5.86e-05 | 0.6766 | yes |
| Next EPS | down | 2.80e-05 | 0.3234 |  |
| Action | BUY | 2.67e-05 | 0.6912 | yes |
| Action | AVOID | 1.19e-05 | 0.3088 |  |

## The two numbers that the scores use

- **p(Next EPS up)** = 5.86e-05 ÷ (5.86e-05 + 2.80e-05) = **0.6766**
- **p(BUY)** = 2.67e-05 ÷ (2.67e-05 + 1.19e-05) = **0.6912**. This is the stock score.

The published file has p(BUY) = 0.6912 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `<|im_start|>` 7.47e-04, `up` 2.54e-04, `�` 1.26e-04, `�` 9.65e-05, `�` 9.50e-05 |
| Pricing | `<|im_start|>` 7.81e-04, `�` 1.11e-04, `flat` 9.77e-05, `�` 8.90e-05, `�` 8.62e-05 |
| Margins | `<|im_start|>` 8.79e-04, `up` 1.17e-04, `�` 1.03e-04, `usr` 8.84e-05, `�` 8.11e-05 |
| Guidance | `<|im_start|>` 8.56e-04, `none` 2.00e-04, `�` 1.09e-04, `held` 1.04e-04, `none` 9.46e-05 |
| Candor | `<|im_start|>` 0.0010, `�` 9.29e-05, `usr` 8.73e-05, `�` 8.53e-05, `�` 8.20e-05 |
| Next EPS | `<|im_start|>` 9.10e-04, `�` 1.02e-04, `usr` 8.94e-05, `�` 8.33e-05, `�` 8.01e-05 |
| Action | `<|im_start|>` 8.84e-04, `�` 1.04e-04, `usr` 8.55e-05, `none` 8.48e-05, `�` 8.48e-05 |
