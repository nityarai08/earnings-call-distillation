# Sample output: sft_read

SFT variant read: read lines and Action, no Next EPS

Call 200826. The model read the prompt in [examples/prompts/200826_rag.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/200826_rag.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: down
Pricing: down
Margins: down
Guidance: cut
Candor: direct
Action: AVOID
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 3.92e-05 | 0.1127 |  |
| Demand | flat | 8.84e-05 | 0.2539 |  |
| Demand | down | 2.20e-04 | 0.6334 | yes |
| Pricing | up | 2.12e-05 | 0.0055 |  |
| Pricing | flat | 1.13e-04 | 0.0296 |  |
| Pricing | down | 0.0037 | 0.9649 | yes |
| Margins | up | 4.12e-05 | 0.0892 |  |
| Margins | flat | 9.64e-05 | 0.2091 |  |
| Margins | down | 3.24e-04 | 0.7017 | yes |
| Guidance | raised | 7.31e-06 | 0.0467 |  |
| Guidance | held | 4.55e-05 | 0.2909 |  |
| Guidance | cut | 7.27e-05 | 0.4648 | yes |
| Guidance | none | 3.09e-05 | 0.1976 |  |
| Candor | direct | 7.20e-05 | 0.8238 | yes |
| Candor | evasive | 1.54e-05 | 0.1762 |  |
| Action | BUY | 8.05e-06 | 0.1566 |  |
| Action | AVOID | 4.34e-05 | 0.8434 | yes |

## The two numbers that the scores use

- **p(BUY)** = 8.05e-06 ÷ (8.05e-06 + 4.34e-05) = **0.1566**. This is the stock score.

The published file has p(BUY) = 0.1566 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `<|im_start|>` 9.13e-04, `down` 2.20e-04, `�` 1.56e-04, `' '` 1.38e-04, `�` 1.32e-04 |
| Pricing | `down` 0.0037, `<|im_start|>` 6.10e-04, `down` 4.98e-04, `-down` 3.82e-04, `�` 3.16e-04 |
| Margins | `<|im_start|>` 8.66e-04, `down` 3.24e-04, `�` 1.60e-04, `�` 1.41e-04, `�` 1.37e-04 |
| Guidance | `<|im_start|>` 9.50e-04, `down` 1.78e-04, `�` 1.62e-04, `�` 1.35e-04, `�` 1.31e-04 |
| Candor | `<|im_start|>` 0.0011, `�` 1.59e-04, `�` 1.51e-04, `�` 1.42e-04, `�` 1.29e-04 |
| Action | `<|im_start|>` 0.0010, `down` 1.85e-04, `�` 1.61e-04, `' '` 1.54e-04, `�` 1.25e-04 |
