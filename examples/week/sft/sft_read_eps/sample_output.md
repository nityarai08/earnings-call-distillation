# Sample output: sft_read_eps

SFT variant read + EPS (the variant kept)

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
| Demand | up | 3.15e-05 | 0.1358 |  |
| Demand | flat | 5.48e-05 | 0.2364 |  |
| Demand | down | 1.46e-04 | 0.6278 | yes |
| Pricing | up | 1.89e-05 | 0.0252 |  |
| Pricing | flat | 3.49e-05 | 0.0466 |  |
| Pricing | down | 6.95e-04 | 0.9282 | yes |
| Margins | up | 3.56e-05 | 0.1190 |  |
| Margins | flat | 5.56e-05 | 0.1857 |  |
| Margins | down | 2.08e-04 | 0.6953 | yes |
| Guidance | raised | 1.07e-05 | 0.0572 |  |
| Guidance | held | 4.21e-05 | 0.2241 |  |
| Guidance | cut | 1.03e-04 | 0.5503 | yes |
| Guidance | none | 3.17e-05 | 0.1685 |  |
| Candor | direct | 3.80e-05 | 0.8158 | yes |
| Candor | evasive | 8.57e-06 | 0.1842 |  |
| Next EPS | up | 3.47e-05 | 0.1277 |  |
| Next EPS | down | 2.37e-04 | 0.8723 | yes |
| Action | BUY | 9.11e-06 | 0.2183 |  |
| Action | AVOID | 3.26e-05 | 0.7817 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 3.47e-05 ÷ (3.47e-05 + 2.37e-04) = **0.1277**
- **p(BUY)** = 9.11e-06 ÷ (9.11e-06 + 3.26e-05) = **0.2183**. This is the stock score.

The published file has p(BUY) = 0.2183 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `<|im_start|>` 8.78e-04, `down` 1.46e-04, `�` 1.03e-04, `�` 8.43e-05, `�` 8.43e-05 |
| Pricing | `<|im_start|>` 7.64e-04, `down` 6.95e-04, `cut` 1.63e-04, `down` 1.31e-04, `�` 1.03e-04 |
| Margins | `<|im_start|>` 8.77e-04, `down` 2.08e-04, `�` 1.02e-04, `�` 8.28e-05, `�` 8.03e-05 |
| Guidance | `<|im_start|>` 8.94e-04, `down` 1.31e-04, `cut` 1.03e-04, `�` 9.72e-05, `�` 8.51e-05 |
| Candor | `<|im_start|>` 9.27e-04, `�` 9.77e-05, `�` 8.49e-05, `�` 8.42e-05, `�` 7.79e-05 |
| Next EPS | `<|im_start|>` 8.67e-04, `down` 2.37e-04, `�` 1.00e-04, `�` 8.19e-05, `cut` 8.19e-05 |
| Action | `<|im_start|>` 8.68e-04, `down` 1.46e-04, `�` 9.74e-05, `�` 7.95e-05, `�` 7.70e-05 |
