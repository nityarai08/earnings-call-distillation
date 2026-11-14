# Sample output: pi_rag_docs

Control: the Pi tool loop, given RAG's documents

Call 200826. The model read the prompt in [examples/prompts/200826_pi_rag_docs.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/200826_pi_rag_docs.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: down
Pricing: down
Margins: down
Guidance: cut
Candor: direct
Next EPS: down
Action: AVOID
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 0.1696 | 0.1714 |  |
| Demand | flat | 0.3590 | 0.3628 |  |
| Demand | down | 0.4609 | 0.4658 | yes |
| Pricing | up | 0.0273 | 0.0274 |  |
| Pricing | flat | 0.0655 | 0.0657 |  |
| Pricing | down | 0.9047 | 0.9069 | yes |
| Margins | up | 0.3177 | 0.3192 |  |
| Margins | flat | 0.3177 | 0.3192 |  |
| Margins | down | 0.3600 | 0.3617 | yes |
| Guidance | raised | 0.0553 | 0.0555 |  |
| Guidance | held | 0.1504 | 0.1509 |  |
| Guidance | cut | 0.6741 | 0.6761 | yes |
| Guidance | none | 0.1171 | 0.1175 |  |
| Candor | direct | 0.8171 | 0.8176 | yes |
| Candor | evasive | 0.1823 | 0.1824 |  |
| Next EPS | up | 0.0203 | 0.0203 |  |
| Next EPS | down | 0.9783 | 0.9797 | yes |
| Action | BUY | 4.84e-04 | 0.0005 |  |
| Action | AVOID | 0.9922 | 0.9995 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.0203 ÷ (0.0203 + 0.9783) = **0.0203**
- **p(BUY)** = 4.84e-04 ÷ (4.84e-04 + 0.9922) = **0.0005**. This is the stock score.

The published file has p(BUY) = 0.0005 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `down` 0.4609, `flat` 0.3590, `up` 0.1696, `cut` 0.0058, `raised` 0.0013 |
| Pricing | `down` 0.9047, `flat` 0.0655, `up` 0.0273, `mixed` 0.0014, `cut` 6.43e-04 |
| Margins | `down` 0.3600, `flat` 0.3177, `up` 0.3177, `cut` 0.0017, `held` 0.0011 |
| Guidance | `cut` 0.6741, `held` 0.1504, `none` 0.1171, `raised` 0.0553, `lowered` 3.29e-04 |
| Candor | `direct` 0.8171, `ev` 0.1823, `e` 6.12e-05, `indirect` 3.95e-05, `evapor` 3.48e-05 |
| Next EPS | `down` 0.9783, `up` 0.0203, `flat` 6.13e-04, `unknown` 4.78e-04, `none` 1.07e-04 |
| Action | `A` 0.9922, `avoid` 0.0036, `AV` 0.0032, `BUY` 4.84e-04, `down` 2.59e-04 |
