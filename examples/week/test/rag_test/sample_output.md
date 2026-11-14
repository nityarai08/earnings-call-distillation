# Sample output: rag_test (week stage)

Final test of the RAG setup.

Call 218058. The model read the prompt in [examples/prompts/218058_rag.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/218058_rag.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: up
Pricing: flat
Margins: up
Guidance: none
Candor: direct
Next EPS: down
Action: AVOID
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 0.7590 | 0.7664 | yes |
| Demand | flat | 0.2175 | 0.2196 |  |
| Demand | down | 0.0139 | 0.0140 |  |
| Pricing | up | 0.1474 | 0.1475 |  |
| Pricing | flat | 0.8482 | 0.8490 | yes |
| Pricing | down | 0.0035 | 0.0035 |  |
| Margins | up | 0.7540 | 0.7546 | yes |
| Margins | flat | 0.2160 | 0.2162 |  |
| Margins | down | 0.0292 | 0.0293 |  |
| Guidance | raised | 0.1369 | 0.1375 |  |
| Guidance | held | 0.3721 | 0.3739 |  |
| Guidance | cut | 0.0647 | 0.0650 |  |
| Guidance | none | 0.4216 | 0.4236 | yes |
| Candor | direct | 0.4990 | 0.5000 | yes |
| Candor | evasive | 0.4990 | 0.5000 |  |
| Next EPS | up | 0.4047 | 0.4073 |  |
| Next EPS | down | 0.5888 | 0.5927 | yes |
| Action | BUY | 0.0286 | 0.0293 |  |
| Action | AVOID | 0.9484 | 0.9707 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.4047 ÷ (0.4047 + 0.5888) = **0.4073**
- **p(BUY)** = 0.0286 ÷ (0.0286 + 0.9484) = **0.0293**. This is the stock score.

The published file has p(BUY) = 0.0293 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `up` 0.7590, `flat` 0.2175, `down` 0.0139, `stable` 0.0066, `Up` 8.89e-04 |
| Pricing | `flat` 0.8482, `up` 0.1474, `down` 0.0035, `fluct` 1.26e-04, `stable` 1.19e-04 |
| Margins | `up` 0.7540, `flat` 0.2160, `down` 0.0292, `held` 1.85e-04, `raised` 1.05e-04 |
| Guidance | `none` 0.4216, `held` 0.3721, `raised` 0.1369, `cut` 0.0647, `None` 3.84e-04 |
| Candor | `direct` 0.4990, `ev` 0.4990, `mixed` 5.84e-04, `e` 1.30e-04, `indirect` 1.15e-04 |
| Next EPS | `down` 0.5888, `up` 0.4047, `flat` 0.0040, `unknown` 0.0015, `none` 2.54e-04 |
| Action | `A` 0.9484, `BUY` 0.0286, `avoid` 0.0105, `AV` 0.0093, `HOLD` 8.65e-04 |
