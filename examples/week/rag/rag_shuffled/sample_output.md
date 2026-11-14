# Sample output: rag_shuffled

Control: the search results, with the past outcomes in them shuffled

Call 200826. The model read the prompt in [examples/prompts/200826_shuffled.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/200826_shuffled.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: down
Pricing: down
Margins: up
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
| Demand | up | 0.0697 | 0.0707 |  |
| Demand | flat | 0.1672 | 0.1695 |  |
| Demand | down | 0.7492 | 0.7598 | yes |
| Pricing | up | 0.0093 | 0.0093 |  |
| Pricing | flat | 0.0416 | 0.0417 |  |
| Pricing | down | 0.9477 | 0.9490 | yes |
| Margins | up | 0.4043 | 0.4055 | yes |
| Margins | flat | 0.2779 | 0.2787 |  |
| Margins | down | 0.3149 | 0.3158 |  |
| Guidance | raised | 0.1734 | 0.1740 |  |
| Guidance | held | 0.1965 | 0.1971 |  |
| Guidance | cut | 0.5341 | 0.5358 | yes |
| Guidance | none | 0.0928 | 0.0931 |  |
| Candor | direct | 0.9769 | 0.9770 | yes |
| Candor | evasive | 0.0230 | 0.0230 |  |
| Next EPS | up | 0.2004 | 0.2018 |  |
| Next EPS | down | 0.7926 | 0.7982 | yes |
| Action | BUY | 0.0022 | 0.0022 |  |
| Action | AVOID | 0.9923 | 0.9978 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.2004 ÷ (0.2004 + 0.7926) = **0.2018**
- **p(BUY)** = 0.0022 ÷ (0.0022 + 0.9923) = **0.0022**. This is the stock score.

The published file has p(BUY) = 0.0022 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `down` 0.7492, `flat` 0.1672, `up` 0.0697, `cut` 0.0107, `Down` 0.0011 |
| Pricing | `down` 0.9477, `flat` 0.0416, `up` 0.0093, `cut` 6.73e-04, `mixed` 4.08e-04 |
| Margins | `up` 0.4043, `down` 0.3149, `flat` 0.2779, `raised` 5.36e-04, `held` 4.73e-04 |
| Guidance | `cut` 0.5341, `held` 0.1965, `raised` 0.1734, `none` 0.0928, `lowered` 4.30e-04 |
| Candor | `direct` 0.9769, `ev` 0.0230, `e` 1.74e-05, `indirect` 9.90e-06, `clear` 9.90e-06 |
| Next EPS | `down` 0.7926, `up` 0.2004, `flat` 0.0047, `unknown` 0.0015, `none` 9.19e-05 |
| Action | `A` 0.9923, `AV` 0.0032, `BUY` 0.0022, `avoid` 0.0019, `down` 2.29e-04 |
