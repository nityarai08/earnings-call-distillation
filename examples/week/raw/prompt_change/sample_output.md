# Sample output: prompt_change

Prompt style: an instruction to compare with the last quarter

Call 102151. The model read the prompt in [examples/prompts/102151_noblock_change.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/102151_noblock_change.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: down
Pricing: flat
Margins: up
Guidance: raised
Candor: direct
Next EPS: up
Action: AVOID
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 0.2770 | 0.2787 |  |
| Demand | flat | 0.3139 | 0.3158 |  |
| Demand | down | 0.4031 | 0.4055 | yes |
| Pricing | up | 0.0104 | 0.0104 |  |
| Pricing | flat | 0.9353 | 0.9367 | yes |
| Pricing | down | 0.0528 | 0.0528 |  |
| Margins | up | 0.9725 | 0.9726 | yes |
| Margins | flat | 0.0178 | 0.0178 |  |
| Margins | down | 0.0095 | 0.0095 |  |
| Guidance | raised | 0.2962 | 0.2986 | yes |
| Guidance | held | 0.2614 | 0.2635 |  |
| Guidance | cut | 0.2036 | 0.2052 |  |
| Guidance | none | 0.2307 | 0.2326 |  |
| Candor | direct | 0.7759 | 0.7773 | yes |
| Candor | evasive | 0.2223 | 0.2227 |  |
| Next EPS | up | 0.6114 | 0.6225 | yes |
| Next EPS | down | 0.3708 | 0.3775 |  |
| Action | BUY | 0.2627 | 0.2689 |  |
| Action | AVOID | 0.7142 | 0.7311 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.6114 ÷ (0.6114 + 0.3708) = **0.6225**
- **p(BUY)** = 0.2627 ÷ (0.2627 + 0.7142) = **0.2689**. This is the stock score.

The published file has p(BUY) = 0.2689 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `down` 0.4031, `flat` 0.3139, `up` 0.2770, `stable` 0.0011, `mixed` 3.68e-04 |
| Pricing | `flat` 0.9353, `down` 0.0528, `up` 0.0104, `none` 4.57e-04, `unknown` 2.44e-04 |
| Margins | `up` 0.9725, `flat` 0.0178, `down` 0.0095, `raised` 3.90e-05, `stable` 1.35e-05 |
| Guidance | `raised` 0.2962, `held` 0.2614, `none` 0.2307, `cut` 0.2036, `lowered` 0.0010 |
| Candor | `direct` 0.7759, `ev` 0.2223, `mixed` 6.65e-04, `mix` 7.01e-05, `indirect` 6.58e-05 |
| Next EPS | `up` 0.6114, `down` 0.3708, `flat` 0.0163, `unknown` 2.63e-04, `not` 1.70e-04 |
| Action | `A` 0.7142, `BUY` 0.2627, `AV` 0.0115, `avoid` 0.0037, `HOLD` 0.0033 |
