# Sample output: rag_test (mini stage)

Final test of the RAG setup.

Call 216962. The model read the prompt in [examples/prompts/216962_rag.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/216962_rag.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: up
Pricing: up
Margins: up
Guidance: raised
Candor: direct
Next EPS: down
Action: AVOID
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 0.9875 | 0.9889 | yes |
| Demand | flat | 0.0075 | 0.0076 |  |
| Demand | down | 0.0036 | 0.0036 |  |
| Pricing | up | 0.9957 | 0.9957 | yes |
| Pricing | flat | 0.0041 | 0.0041 |  |
| Pricing | down | 2.30e-04 | 0.0002 |  |
| Margins | up | 0.9400 | 0.9406 | yes |
| Margins | flat | 0.0530 | 0.0531 |  |
| Margins | down | 0.0063 | 0.0063 |  |
| Guidance | raised | 0.7611 | 0.7625 | yes |
| Guidance | held | 0.1323 | 0.1325 |  |
| Guidance | cut | 0.0139 | 0.0140 |  |
| Guidance | none | 0.0909 | 0.0911 |  |
| Candor | direct | 0.9577 | 0.9579 | yes |
| Candor | evasive | 0.0421 | 0.0421 |  |
| Next EPS | up | 0.3757 | 0.3775 |  |
| Next EPS | down | 0.6194 | 0.6225 | yes |
| Action | BUY | 0.3172 | 0.3208 |  |
| Action | AVOID | 0.6715 | 0.6792 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.3757 ÷ (0.3757 + 0.6194) = **0.3775**
- **p(BUY)** = 0.3172 ÷ (0.3172 + 0.6715) = **0.3208**. This is the stock score.

The published file has p(BUY) = 0.3208 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `up` 0.9875, `flat` 0.0075, `down` 0.0036, `raised` 7.01e-04, `Up` 2.92e-04 |
| Pricing | `up` 0.9957, `flat` 0.0041, `down` 2.30e-04, `raised` 4.76e-06, `Up` 4.48e-06 |
| Margins | `up` 0.9400, `flat` 0.0530, `down` 0.0063, `held` 4.05e-04, `raised` 1.16e-04 |
| Guidance | `raised` 0.7611, `held` 0.1323, `none` 0.0909, `cut` 0.0139, `None` 1.37e-04 |
| Candor | `direct` 0.9577, `ev` 0.0421, `indirect` 1.81e-05, `clear` 1.25e-05, `e` 8.56e-06 |
| Next EPS | `down` 0.6194, `up` 0.3757, `flat` 0.0029, `unknown` 3.88e-04, `cut` 3.43e-04 |
| Action | `A` 0.6715, `BUY` 0.3172, `AV` 0.0066, `HOLD` 0.0017, `avoid` 0.0013 |
