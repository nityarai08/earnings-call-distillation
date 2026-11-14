# Sample output: pi

Pi setup: the model searches the library itself

Call 200826. The model read the prompt in [examples/prompts/200826_pi.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/200826_pi.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: down
Pricing: down
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
| Demand | up | 0.3025 | 0.3045 |  |
| Demand | flat | 0.3025 | 0.3045 |  |
| Demand | down | 0.3884 | 0.3910 | yes |
| Pricing | up | 0.0557 | 0.0558 |  |
| Pricing | flat | 0.0715 | 0.0716 |  |
| Pricing | down | 0.8708 | 0.8726 | yes |
| Margins | up | 0.5856 | 0.5871 | yes |
| Margins | flat | 0.2441 | 0.2447 |  |
| Margins | down | 0.1678 | 0.1682 |  |
| Guidance | raised | 0.3429 | 0.3441 | yes |
| Guidance | held | 0.3026 | 0.3037 |  |
| Guidance | cut | 0.2080 | 0.2087 |  |
| Guidance | none | 0.1430 | 0.1435 |  |
| Candor | direct | 0.9769 | 0.9770 | yes |
| Candor | evasive | 0.0230 | 0.0230 |  |
| Next EPS | up | 0.1804 | 0.1824 |  |
| Next EPS | down | 0.8087 | 0.8176 | yes |
| Action | BUY | 0.0366 | 0.0373 |  |
| Action | AVOID | 0.9449 | 0.9627 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.1804 ÷ (0.1804 + 0.8087) = **0.1824**
- **p(BUY)** = 0.0366 ÷ (0.0366 + 0.9449) = **0.0373**. This is the stock score.

The published file has p(BUY) = 0.0373 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `down` 0.3884, `flat` 0.3025, `up` 0.3025, `cut` 0.0018, `raised` 0.0010 |
| Pricing | `down` 0.8708, `flat` 0.0715, `up` 0.0557, `mixed` 0.0012, `cut` 4.25e-04 |
| Margins | `up` 0.5856, `flat` 0.2441, `down` 0.1678, `held` 6.86e-04, `raised` 3.67e-04 |
| Guidance | `raised` 0.3429, `held` 0.3026, `cut` 0.2080, `none` 0.1430, `lowered` 1.90e-04 |
| Candor | `direct` 0.9769, `ev` 0.0230, `indirect` 1.63e-05, `e` 1.12e-05, `clear` 8.73e-06 |
| Next EPS | `down` 0.8087, `up` 0.1804, `unknown` 0.0062, `flat` 0.0029, `none` 7.37e-04 |
| Action | `A` 0.9449, `BUY` 0.0366, `AV` 0.0093, `avoid` 0.0072, `up` 4.61e-04 |
