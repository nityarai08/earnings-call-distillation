# Sample output: rag_random

Control: random passages of the same length instead of the search results

Call 200826. The model read the prompt in [examples/prompts/200826_random.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/200826_random.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: down
Pricing: down
Margins: flat
Guidance: cut
Candor: evasive
Next EPS: down
Action: AVOID
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 0.0626 | 0.0629 |  |
| Demand | flat | 0.1701 | 0.1710 |  |
| Demand | down | 0.7623 | 0.7662 | yes |
| Pricing | up | 0.0188 | 0.0188 |  |
| Pricing | flat | 0.0744 | 0.0744 |  |
| Pricing | down | 0.9059 | 0.9068 | yes |
| Margins | up | 0.1165 | 0.1168 |  |
| Margins | flat | 0.5219 | 0.5234 | yes |
| Margins | down | 0.3587 | 0.3598 |  |
| Guidance | raised | 0.0291 | 0.0292 |  |
| Guidance | held | 0.0897 | 0.0900 |  |
| Guidance | cut | 0.6626 | 0.6649 | yes |
| Guidance | none | 0.2151 | 0.2159 |  |
| Candor | direct | 0.3203 | 0.3208 |  |
| Candor | evasive | 0.6782 | 0.6792 | yes |
| Next EPS | up | 0.0110 | 0.0110 |  |
| Next EPS | down | 0.9871 | 0.9890 | yes |
| Action | BUY | 7.45e-05 | 0.0001 |  |
| Action | AVOID | 0.9954 | 0.9999 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.0110 ÷ (0.0110 + 0.9871) = **0.0110**
- **p(BUY)** = 7.45e-05 ÷ (7.45e-05 + 0.9954) = **0.0001**. This is the stock score.

The published file has p(BUY) = 0.0001 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `down` 0.7623, `flat` 0.1701, `up` 0.0626, `cut` 0.0019, `Down` 7.88e-04 |
| Pricing | `down` 0.9059, `flat` 0.0744, `up` 0.0188, `cut` 3.90e-04, `mixed` 1.73e-04 |
| Margins | `flat` 0.5219, `down` 0.3587, `up` 0.1165, `cut` 0.0013, `held` 4.20e-04 |
| Guidance | `cut` 0.6626, `none` 0.2151, `held` 0.0897, `raised` 0.0291, `lowered` 2.09e-04 |
| Candor | `ev` 0.6782, `direct` 0.3203, `mixed` 1.14e-04, `e` 7.86e-05, `evapor` 7.86e-05 |
| Next EPS | `down` 0.9871, `up` 0.0110, `flat` 0.0010, `cut` 4.82e-04, `unknown` 1.77e-04 |
| Action | `A` 0.9954, `avoid` 0.0025, `AV` 0.0019, `Avoid` 8.44e-05, `BUY` 7.45e-05 |
