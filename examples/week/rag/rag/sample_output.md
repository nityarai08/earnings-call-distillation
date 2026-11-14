# Sample output: rag

RAG setup: a fixed search adds a knowledge block

Call 200826. The model read the prompt in [examples/prompts/200826_rag.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/200826_rag.txt). The prompt ends with the label `Demand:`.

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
| Demand | up | 0.0765 | 0.0776 |  |
| Demand | flat | 0.1835 | 0.1862 |  |
| Demand | down | 0.7258 | 0.7362 | yes |
| Pricing | up | 0.0082 | 0.0082 |  |
| Pricing | flat | 0.0470 | 0.0470 |  |
| Pricing | down | 0.9436 | 0.9448 | yes |
| Margins | up | 0.2988 | 0.2996 |  |
| Margins | flat | 0.2637 | 0.2644 |  |
| Margins | down | 0.4348 | 0.4360 | yes |
| Guidance | raised | 0.0337 | 0.0337 |  |
| Guidance | held | 0.1175 | 0.1177 |  |
| Guidance | cut | 0.7659 | 0.7676 | yes |
| Guidance | none | 0.0807 | 0.0809 |  |
| Candor | direct | 0.9145 | 0.9149 | yes |
| Candor | evasive | 0.0851 | 0.0851 |  |
| Next EPS | up | 0.0229 | 0.0230 |  |
| Next EPS | down | 0.9756 | 0.9770 | yes |
| Action | BUY | 8.45e-05 | 0.0001 |  |
| Action | AVOID | 0.9967 | 0.9999 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 0.0229 ÷ (0.0229 + 0.9756) = **0.0230**
- **p(BUY)** = 8.45e-05 ÷ (8.45e-05 + 0.9967) = **0.0001**. This is the stock score.

The published file has p(BUY) = 0.0001 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `down` 0.7258, `flat` 0.1835, `up` 0.0765, `cut` 0.0104, `Down` 0.0014 |
| Pricing | `down` 0.9436, `flat` 0.0470, `up` 0.0082, `mixed` 5.22e-04, `cut` 4.61e-04 |
| Margins | `down` 0.4348, `up` 0.2988, `flat` 0.2637, `cut` 7.41e-04, `held` 3.72e-04 |
| Guidance | `cut` 0.7659, `held` 0.1175, `none` 0.0807, `raised` 0.0337, `down` 2.41e-04 |
| Candor | `direct` 0.9145, `ev` 0.0851, `e` 4.42e-05, `clear` 2.22e-05, `indirect` 2.09e-05 |
| Next EPS | `down` 0.9756, `up` 0.0229, `flat` 8.90e-04, `unknown` 3.27e-04, `cut` 5.02e-05 |
| Action | `A` 0.9967, `AV` 0.0017, `avoid` 0.0013, `BUY` 8.45e-05, `Avoid` 5.81e-05 |
