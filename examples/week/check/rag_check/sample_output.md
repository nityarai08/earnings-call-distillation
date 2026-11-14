# Sample output: rag_check

RAG setup on the check calls

Call 62273. The model read the prompt in [examples/prompts/62273_rag_noreaction.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/62273_rag_noreaction.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: down
Pricing: flat
Margins: flat
Guidance: held
Candor: evasive
Next EPS: down
Action: AVOID
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 0.3051 | 0.3344 |  |
| Demand | flat | 0.1633 | 0.1790 |  |
| Demand | down | 0.4439 | 0.4866 | yes |
| Pricing | up | 0.0372 | 0.0373 |  |
| Pricing | flat | 0.8460 | 0.8480 | yes |
| Pricing | down | 0.1145 | 0.1148 |  |
| Margins | up | 0.2702 | 0.2713 |  |
| Margins | flat | 0.6483 | 0.6509 | yes |
| Margins | down | 0.0774 | 0.0777 |  |
| Guidance | raised | 0.0474 | 0.0476 |  |
| Guidance | held | 0.6542 | 0.6569 | yes |
| Guidance | cut | 0.1288 | 0.1294 |  |
| Guidance | none | 0.1654 | 0.1661 |  |
| Candor | direct | 0.1817 | 0.1824 |  |
| Candor | evasive | 0.8144 | 0.8176 | yes |
| Next EPS | up | 0.0159 | 0.0159 |  |
| Next EPS | down | 0.9833 | 0.9841 | yes |
| Action | BUY | 8.36e-05 | 0.0001 |  |
| Action | AVOID | 0.9862 | 0.9999 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.0159 ÷ (0.0159 + 0.9833) = **0.0159**
- **p(BUY)** = 8.36e-05 ÷ (8.36e-05 + 0.9862) = **0.0001**. This is the stock score.

The published file has p(BUY) = 0.0001 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `down` 0.4439, `up` 0.3051, `flat` 0.1633, `mixed` 0.0771, `held` 0.0013 |
| Pricing | `flat` 0.8460, `down` 0.1145, `up` 0.0372, `unknown` 0.0011, `none` 3.22e-04 |
| Margins | `flat` 0.6483, `up` 0.2702, `down` 0.0774, `held` 0.0030, `raised` 2.79e-04 |
| Guidance | `held` 0.6542, `none` 0.1654, `cut` 0.1288, `raised` 0.0474, `None` 2.34e-04 |
| Candor | `ev` 0.8144, `direct` 0.1817, `mixed` 0.0020, `cautious` 1.56e-04, `evapor` 1.29e-04 |
| Next EPS | `down` 0.9833, `up` 0.0159, `flat` 6.16e-04, `unknown` 2.39e-05, `cut` 1.64e-05 |
| Action | `A` 0.9862, `avoid` 0.0075, `AV` 0.0059, `Avoid` 2.01e-04, `BUY` 8.36e-05 |
