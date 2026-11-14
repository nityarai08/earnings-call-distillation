# Sample output: raw_test (mini stage)

Final test of the Raw setup.

Call 216962. The model read the prompt in [examples/prompts/216962_noblock.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/216962_noblock.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: up
Pricing: up
Margins: up
Guidance: raised
Candor: direct
Next EPS: down
Action: BUY
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 0.9937 | 0.9945 | yes |
| Demand | flat | 0.0046 | 0.0046 |  |
| Demand | down | 9.06e-04 | 0.0009 |  |
| Pricing | up | 0.9944 | 0.9944 | yes |
| Pricing | flat | 0.0052 | 0.0052 |  |
| Pricing | down | 3.34e-04 | 0.0003 |  |
| Margins | up | 0.9527 | 0.9531 | yes |
| Margins | flat | 0.0419 | 0.0419 |  |
| Margins | down | 0.0050 | 0.0050 |  |
| Guidance | raised | 0.7502 | 0.7517 | yes |
| Guidance | held | 0.1304 | 0.1306 |  |
| Guidance | cut | 0.0024 | 0.0024 |  |
| Guidance | none | 0.1150 | 0.1153 |  |
| Candor | direct | 0.9463 | 0.9466 | yes |
| Candor | evasive | 0.0534 | 0.0534 |  |
| Next EPS | up | 0.2402 | 0.2451 |  |
| Next EPS | down | 0.7398 | 0.7549 | yes |
| Action | BUY | 0.7644 | 0.7773 | yes |
| Action | AVOID | 0.2190 | 0.2227 |  |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.2402 ÷ (0.2402 + 0.7398) = **0.2451**
- **p(BUY)** = 0.7644 ÷ (0.7644 + 0.2190) = **0.7773**. This is the stock score.

The published file has p(BUY) = 0.7773 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `up` 0.9937, `flat` 0.0046, `down` 9.06e-04, `Up` 2.60e-04, `raised` 2.02e-04 |
| Pricing | `up` 0.9944, `flat` 0.0052, `down` 3.34e-04, `raised` 1.47e-05, `held` 5.07e-06 |
| Margins | `up` 0.9527, `flat` 0.0419, `down` 0.0050, `none` 1.04e-04, `held` 8.08e-05 |
| Guidance | `raised` 0.7502, `held` 0.1304, `none` 0.1150, `cut` 0.0024, `None` 1.05e-04 |
| Candor | `direct` 0.9463, `ev` 0.0534, `mixed` 2.95e-05, `indirect` 2.45e-05, `clear` 2.30e-05 |
| Next EPS | `down` 0.7398, `up` 0.2402, `unknown` 0.0093, `flat` 0.0056, `none` 0.0034 |
| Action | `BUY` 0.7644, `A` 0.2190, `HOLD` 0.0085, `AV` 0.0035, `buy` 7.90e-04 |
