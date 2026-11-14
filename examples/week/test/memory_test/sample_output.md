# Sample output: memory_test (week stage)

Final test of the Memory only setup.

Call 218058. The model read the prompt in [examples/prompts/218058_nocall_noblock.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/218058_nocall_noblock.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: down
Pricing: flat
Margins: flat
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
| Demand | up | 0.1255 | 0.1258 |  |
| Demand | flat | 0.2345 | 0.2351 |  |
| Demand | down | 0.6375 | 0.6391 | yes |
| Pricing | up | 0.0391 | 0.0391 |  |
| Pricing | flat | 0.5400 | 0.5402 | yes |
| Pricing | down | 0.4205 | 0.4207 |  |
| Margins | up | 0.0996 | 0.0997 |  |
| Margins | flat | 0.5059 | 0.5061 | yes |
| Margins | down | 0.3940 | 0.3942 |  |
| Guidance | raised | 8.37e-05 | 0.0001 |  |
| Guidance | held | 0.0067 | 0.0067 |  |
| Guidance | cut | 0.0046 | 0.0046 |  |
| Guidance | none | 0.9873 | 0.9887 | yes |
| Candor | direct | 0.3021 | 0.3486 |  |
| Candor | evasive | 0.5644 | 0.6514 | yes |
| Next EPS | up | 0.4635 | 0.4688 |  |
| Next EPS | down | 0.5252 | 0.5312 | yes |
| Action | BUY | 2.02e-04 | 0.0002 |  |
| Action | AVOID | 0.9933 | 0.9998 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.4635 ÷ (0.4635 + 0.5252) = **0.4688**
- **p(BUY)** = 2.02e-04 ÷ (2.02e-04 + 0.9933) = **0.0002**. This is the stock score.

The published file has p(BUY) = 0.0002 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `down` 0.6375, `flat` 0.2345, `up` 0.1255, `none` 4.25e-04, `Down` 4.25e-04 |
| Pricing | `flat` 0.5400, `down` 0.4205, `up` 0.0391, `neutral` 3.80e-05, `fl` 3.35e-05 |
| Margins | `flat` 0.5059, `down` 0.3940, `up` 0.0996, `fl` 5.18e-05, `stable` 4.29e-05 |
| Guidance | `none` 0.9873, `held` 0.0067, `cut` 0.0046, `no` 3.11e-04, `None` 2.14e-04 |
| Candor | `ev` 0.5644, `direct` 0.3021, `none` 0.1111, `not` 0.0103, `unknown` 0.0049 |
| Next EPS | `down` 0.5252, `up` 0.4635, `flat` 0.0085, `unknown` 0.0019, `none` 1.46e-04 |
| Action | `A` 0.9933, `avoid` 0.0046, `AV` 0.0017, `BUY` 2.02e-04, `Avoid` 1.23e-04 |
