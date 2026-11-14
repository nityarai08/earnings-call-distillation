# Sample output: raw_long

Position check: the model holds the stock and chooses HOLD or SELL

Call 102151. The model read the prompt in [examples/prompts/102151_noblock_holds.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/102151_noblock_holds.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: up
Pricing: flat
Margins: up
Guidance: raised
Candor: direct
Next EPS: up
Action: HOLD
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 0.3586 | 0.3599 | yes |
| Demand | flat | 0.2793 | 0.2803 |  |
| Demand | down | 0.3586 | 0.3599 |  |
| Pricing | up | 0.0329 | 0.0329 |  |
| Pricing | flat | 0.9613 | 0.9620 | yes |
| Pricing | down | 0.0050 | 0.0050 |  |
| Margins | up | 0.9982 | 0.9982 | yes |
| Margins | flat | 0.0015 | 0.0015 |  |
| Margins | down | 2.96e-04 | 0.0003 |  |
| Guidance | raised | 0.5297 | 0.5322 | yes |
| Guidance | held | 0.2502 | 0.2514 |  |
| Guidance | cut | 0.0435 | 0.0437 |  |
| Guidance | none | 0.1720 | 0.1728 |  |
| Candor | direct | 0.9040 | 0.9047 | yes |
| Candor | evasive | 0.0953 | 0.0953 |  |
| Next EPS | up | 0.8859 | 0.8933 | yes |
| Next EPS | down | 0.1058 | 0.1067 |  |
| Action | HOLD | 0.8743 | 0.9579 | yes |
| Action | SELL | 0.0384 | 0.0421 |  |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.8859 ÷ (0.8859 + 0.1058) = **0.8933**
- **p(HOLD)** = 0.8743 ÷ (0.8743 + 0.0384) = **0.9579**. This is the stock score.

The published file has p(HOLD) = 0.9579 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `up` 0.3586, `down` 0.3586, `flat` 0.2793, `stable` 5.39e-04, `Flat` 2.55e-04 |
| Pricing | `flat` 0.9613, `up` 0.0329, `down` 0.0050, `held` 2.85e-04, `unknown` 7.66e-05 |
| Margins | `up` 0.9982, `flat` 0.0015, `down` 2.96e-04, `Up` 1.37e-06, `'\n'` 1.07e-06 |
| Guidance | `raised` 0.5297, `held` 0.2502, `none` 0.1720, `cut` 0.0435, `None` 2.59e-04 |
| Candor | `direct` 0.9040, `ev` 0.0953, `mixed` 2.68e-04, `mix` 3.62e-05, `indirect` 3.40e-05 |
| Next EPS | `up` 0.8859, `down` 0.1058, `flat` 0.0077, `not` 1.09e-04, `raised` 5.85e-05 |
| Action | `HOLD` 0.8743, `BUY` 0.0813, `S` 0.0384, `buy` 0.0032, `hold` 7.49e-04 |
