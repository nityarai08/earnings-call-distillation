# Sample output: sft_best_test (mini stage)

Final test of the SFT setup.

Call 216962. The model read the prompt in [examples/prompts/216962_rag.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/216962_rag.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: up
Pricing: up
Margins: up
Guidance: held
Candor: direct
Next EPS: down
Action: BUY
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 2.29e-04 | 0.8328 | yes |
| Demand | flat | 3.10e-05 | 0.1127 |  |
| Demand | down | 1.50e-05 | 0.0545 |  |
| Pricing | up | 1.24e-04 | 0.6141 | yes |
| Pricing | flat | 5.07e-05 | 0.2520 |  |
| Pricing | down | 2.70e-05 | 0.1339 |  |
| Margins | up | 1.15e-04 | 0.5492 | yes |
| Margins | flat | 5.69e-05 | 0.2719 |  |
| Margins | down | 3.75e-05 | 0.1790 |  |
| Guidance | raised | 1.03e-05 | 0.0885 |  |
| Guidance | held | 4.53e-05 | 0.3875 | yes |
| Guidance | cut | 2.61e-05 | 0.2234 |  |
| Guidance | none | 3.51e-05 | 0.3006 |  |
| Candor | direct | 4.16e-05 | 0.7349 | yes |
| Candor | evasive | 1.50e-05 | 0.2651 |  |
| Next EPS | up | 7.50e-05 | 0.4513 |  |
| Next EPS | down | 9.12e-05 | 0.5487 | yes |
| Action | BUY | 3.22e-05 | 0.7033 | yes |
| Action | AVOID | 1.36e-05 | 0.2967 |  |

## The two numbers that the scores use

- **p(Next EPS up)** = 7.50e-05 ÷ (7.50e-05 + 9.12e-05) = **0.4513**
- **p(BUY)** = 3.22e-05 ÷ (3.22e-05 + 1.36e-05) = **0.7033**. This is the stock score.

The published file has p(BUY) = 0.7034 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `<|im_start|>` 9.35e-04, `' '` 2.64e-04, `up` 2.29e-04, `.` 2.26e-04, `-` 2.26e-04 |
| Pricing | `<|im_start|>` 9.43e-04, `' '` 1.89e-04, `-` 1.86e-04, `�` 1.83e-04, `.` 1.74e-04 |
| Margins | `<|im_start|>` 8.90e-04, `-` 1.81e-04, `�` 1.78e-04, `' '` 1.73e-04, `.` 1.70e-04 |
| Guidance | `<|im_start|>` 9.02e-04, `�` 1.89e-04, `�` 1.89e-04, `�` 1.64e-04, `�` 1.59e-04 |
| Candor | `<|im_start|>` 7.90e-04, `�` 1.94e-04, `�` 1.76e-04, `�` 1.74e-04, `-` 1.66e-04 |
| Next EPS | `<|im_start|>` 8.06e-04, `�` 1.83e-04, `�` 1.80e-04, `-` 1.77e-04, `.` 1.66e-04 |
| Action | `<|im_start|>` 7.77e-04, `' '` 1.71e-04, `�` 1.65e-04, `-` 1.65e-04, `�` 1.63e-04 |
