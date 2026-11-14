# Sample output: prompt_plain

Prompt style: the plain instruction

Call 102151. The model read the prompt in [examples/prompts/102151_noblock.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/102151_noblock.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: up
Pricing: flat
Margins: up
Guidance: raised
Candor: direct
Next EPS: up
Action: BUY
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 0.4483 | 0.4499 | yes |
| Demand | flat | 0.2399 | 0.2408 |  |
| Demand | down | 0.3081 | 0.3092 |  |
| Pricing | up | 0.0329 | 0.0330 |  |
| Pricing | flat | 0.9629 | 0.9636 | yes |
| Pricing | down | 0.0035 | 0.0035 |  |
| Margins | up | 0.9982 | 0.9982 | yes |
| Margins | flat | 0.0015 | 0.0015 |  |
| Margins | down | 2.96e-04 | 0.0003 |  |
| Guidance | raised | 0.4956 | 0.4979 | yes |
| Guidance | held | 0.2653 | 0.2665 |  |
| Guidance | cut | 0.0522 | 0.0525 |  |
| Guidance | none | 0.1823 | 0.1832 |  |
| Candor | direct | 0.9318 | 0.9325 | yes |
| Candor | evasive | 0.0675 | 0.0675 |  |
| Next EPS | up | 0.8590 | 0.8670 | yes |
| Next EPS | down | 0.1317 | 0.1330 |  |
| Action | BUY | 0.7474 | 0.7549 | yes |
| Action | AVOID | 0.2427 | 0.2451 |  |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.8590 ÷ (0.8590 + 0.1317) = **0.8670**
- **p(BUY)** = 0.7474 ÷ (0.7474 + 0.2427) = **0.7549**. This is the stock score.

The published file has p(BUY) = 0.7549 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `up` 0.4483, `down` 0.3081, `flat` 0.2399, `stable` 6.33e-04, `Flat` 2.48e-04 |
| Pricing | `flat` 0.9629, `up` 0.0329, `down` 0.0035, `held` 1.96e-04, `none` 1.53e-04 |
| Margins | `up` 0.9982, `flat` 0.0015, `down` 2.96e-04, `Up` 1.29e-06, `'\n'` 9.41e-07 |
| Guidance | `raised` 0.4956, `held` 0.2653, `none` 0.1823, `cut` 0.0522, `None` 2.58e-04 |
| Candor | `direct` 0.9318, `ev` 0.0675, `mixed` 2.76e-04, `mix` 3.51e-05, `indirect` 2.73e-05 |
| Next EPS | `up` 0.8590, `down` 0.1317, `flat` 0.0084, `not` 1.45e-04, `none` 1.28e-04 |
| Action | `BUY` 0.7474, `A` 0.2427, `AV` 0.0044, `buy` 0.0016, `HOLD` 8.75e-04 |
