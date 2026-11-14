# Sample output: prompt_habit

Prompt style: an instruction to weigh specifics over positive tone

Call 102151. The model read the prompt in [examples/prompts/102151_noblock_habit.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/102151_noblock_habit.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: flat
Pricing: flat
Margins: up
Guidance: none
Candor: evasive
Next EPS: down
Action: AVOID
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 0.0892 | 0.0896 |  |
| Demand | flat | 0.4531 | 0.4552 | yes |
| Demand | down | 0.4531 | 0.4552 |  |
| Pricing | up | 0.0221 | 0.0221 |  |
| Pricing | flat | 0.9397 | 0.9414 | yes |
| Pricing | down | 0.0364 | 0.0365 |  |
| Margins | up | 0.9517 | 0.9520 | yes |
| Margins | flat | 0.0326 | 0.0326 |  |
| Margins | down | 0.0154 | 0.0154 |  |
| Guidance | raised | 0.1251 | 0.1261 |  |
| Guidance | held | 0.3401 | 0.3427 |  |
| Guidance | cut | 0.1418 | 0.1429 |  |
| Guidance | none | 0.3854 | 0.3883 | yes |
| Candor | direct | 0.0032 | 0.0032 |  |
| Candor | evasive | 0.9964 | 0.9968 | yes |
| Next EPS | up | 0.3391 | 0.3486 |  |
| Next EPS | down | 0.6336 | 0.6514 | yes |
| Action | BUY | 5.69e-05 | 0.0001 |  |
| Action | AVOID | 0.9758 | 0.9999 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.3391 ÷ (0.3391 + 0.6336) = **0.3486**
- **p(BUY)** = 5.69e-05 ÷ (5.69e-05 + 0.9758) = **0.0001**. This is the stock score.

The published file has p(BUY) = 0.0001 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `flat` 0.4531, `down` 0.4531, `up` 0.0892, `stable` 9.31e-04, `Flat` 3.88e-04 |
| Pricing | `flat` 0.9397, `down` 0.0364, `up` 0.0221, `unknown` 6.67e-04, `none` 1.80e-04 |
| Margins | `up` 0.9517, `flat` 0.0326, `down` 0.0154, `mixed` 3.81e-05, `raised` 2.62e-05 |
| Guidance | `none` 0.3854, `held` 0.3401, `cut` 0.1418, `raised` 0.1251, `lowered` 4.80e-04 |
| Candor | `ev` 0.9964, `direct` 0.0032, `mixed` 7.46e-05, `cautious` 3.11e-05, `vague` 2.58e-05 |
| Next EPS | `down` 0.6336, `up` 0.3391, `flat` 0.0217, `unknown` 0.0023, `none` 0.0011 |
| Action | `A` 0.9758, `avoid` 0.0179, `AV` 0.0058, `Avoid` 2.25e-04, `BUY` 5.69e-05 |
