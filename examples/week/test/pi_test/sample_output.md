# Sample output: pi_test (week stage)

Final test of the Pi setup.

Call 218058. The model read the prompt in [examples/prompts/218058_pi.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/218058_pi.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: flat
Pricing: flat
Margins: up
Guidance: none
Candor: evasive
Next EPS: up
Action: AVOID
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 0.3941 | 0.4017 |  |
| Demand | flat | 0.5734 | 0.5845 | yes |
| Demand | down | 0.0135 | 0.0137 |  |
| Pricing | up | 0.3183 | 0.3187 |  |
| Pricing | flat | 0.6738 | 0.6747 | yes |
| Pricing | down | 0.0066 | 0.0066 |  |
| Margins | up | 0.6648 | 0.6655 | yes |
| Margins | flat | 0.3140 | 0.3144 |  |
| Margins | down | 0.0201 | 0.0201 |  |
| Guidance | raised | 0.0530 | 0.0533 |  |
| Guidance | held | 0.3916 | 0.3939 |  |
| Guidance | cut | 0.0468 | 0.0470 |  |
| Guidance | none | 0.5028 | 0.5058 | yes |
| Candor | direct | 0.4371 | 0.4378 |  |
| Candor | evasive | 0.5612 | 0.5622 | yes |
| Next EPS | up | 0.8550 | 0.8670 | yes |
| Next EPS | down | 0.1311 | 0.1330 |  |
| Action | BUY | 0.0716 | 0.0759 |  |
| Action | AVOID | 0.8725 | 0.9241 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.8550 ÷ (0.8550 + 0.1311) = **0.8670**
- **p(BUY)** = 0.0716 ÷ (0.0716 + 0.8725) = **0.0759**. This is the stock score.

The published file has p(BUY) = 0.0759 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `flat` 0.5734, `up` 0.3941, `stable` 0.0153, `down` 0.0135, `Flat` 0.0013 |
| Pricing | `flat` 0.6738, `up` 0.3183, `down` 0.0066, `stable` 2.73e-04, `unknown` 1.46e-04 |
| Margins | `up` 0.6648, `flat` 0.3140, `down` 0.0201, `held` 1.35e-04, `raised` 1.19e-04 |
| Guidance | `none` 0.5028, `held` 0.3916, `raised` 0.0530, `cut` 0.0468, `None` 4.31e-04 |
| Candor | `ev` 0.5612, `direct` 0.4371, `mixed` 2.57e-04, `e` 1.38e-04, `indirect` 1.22e-04 |
| Next EPS | `up` 0.8550, `down` 0.1311, `flat` 0.0122, `none` 6.07e-04, `raised` 1.20e-04 |
| Action | `A` 0.8725, `BUY` 0.0716, `avoid` 0.0233, `AV` 0.0141, `HOLD` 0.0067 |
