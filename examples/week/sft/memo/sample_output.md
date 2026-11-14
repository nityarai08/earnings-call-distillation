# Sample output: memo

Memorization check: 50 training calls, trained 10 times, then read back

Call 60332. The model read the prompt in [examples/prompts/60332_rag.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/60332_rag.txt). The prompt ends with the label `Demand:`.

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
| Demand | up | 0.0521 | 0.9861 | yes |
| Demand | flat | 7.31e-04 | 0.0138 |  |
| Demand | down | 3.18e-06 | 0.0001 |  |
| Pricing | up | 0.0017 | 0.3774 |  |
| Pricing | flat | 0.0028 | 0.6223 | yes |
| Pricing | down | 1.35e-06 | 0.0003 |  |
| Margins | up | 0.0022 | 0.6280 | yes |
| Margins | flat | 9.83e-04 | 0.2787 |  |
| Margins | down | 3.29e-04 | 0.0933 |  |
| Guidance | raised | 3.68e-05 | 0.0020 |  |
| Guidance | held | 2.77e-04 | 0.0152 |  |
| Guidance | cut | 7.41e-07 | 0.0000 |  |
| Guidance | none | 0.0179 | 0.9828 | yes |
| Candor | direct | 0.0056 | 0.9334 | yes |
| Candor | evasive | 3.99e-04 | 0.0666 |  |
| Next EPS | up | 0.0014 | 0.9870 | yes |
| Next EPS | down | 1.79e-05 | 0.0130 |  |
| Action | BUY | 4.15e-04 | 0.8311 | yes |
| Action | AVOID | 8.43e-05 | 0.1689 |  |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.0014 ÷ (0.0014 + 1.79e-05) = **0.9870**
- **p(BUY)** = 4.15e-04 ÷ (4.15e-04 + 8.43e-05) = **0.8311**. This is the stock score.

The published file has p(BUY) = 0.8311 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `up` 0.0521, `-up` 0.0017, `up` 0.0012, `Up` 9.54e-04, `direct` 7.78e-04 |
| Pricing | `flat` 0.0028, `up` 0.0017, `,` 7.66e-04, `' '` 5.88e-04, `in` 4.95e-04 |
| Margins | `up` 0.0022, `flat` 9.83e-04, `,` 5.43e-04, `' '` 5.26e-04, `in` 4.72e-04 |
| Guidance | `none` 0.0179, `flat` 0.0041, `none` 0.0017, `None` 8.93e-04, `no` 8.65e-04 |
| Candor | `direct` 0.0056, `flat` 0.0016, `directly` 7.81e-04, `,` 7.56e-04, `' '` 6.47e-04 |
| Next EPS | `up` 0.0014, `flat` 0.0012, `none` 9.94e-04, `' '` 6.41e-04, `,` 5.84e-04 |
| Action | `buy` 6.42e-04, `flat` 6.32e-04, `<|im_start|>` 5.24e-04, `purchase` 4.28e-04, `BUY` 4.15e-04 |
