# Sample output: sft_placebo

Control: SFT on Opus answers shuffled among calls of the same month

Call 200826. The model read the prompt in [examples/prompts/200826_rag.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/200826_rag.txt). The prompt ends with the label `Demand:`.

## What the model wrote

```
Demand: up
Pricing: up
Margins: up
Guidance: none
Candor: direct
Next EPS: up
Action: AVOID
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Demand | up | 8.24e-05 | 0.4678 | yes |
| Demand | flat | 4.38e-05 | 0.2485 |  |
| Demand | down | 5.00e-05 | 0.2837 |  |
| Pricing | up | 5.12e-05 | 0.3662 | yes |
| Pricing | flat | 4.45e-05 | 0.3181 |  |
| Pricing | down | 4.41e-05 | 0.3157 |  |
| Margins | up | 5.89e-05 | 0.4173 | yes |
| Margins | flat | 4.05e-05 | 0.2868 |  |
| Margins | down | 4.18e-05 | 0.2959 |  |
| Guidance | raised | 1.25e-05 | 0.0917 |  |
| Guidance | held | 4.01e-05 | 0.2937 |  |
| Guidance | cut | 1.52e-05 | 0.1110 |  |
| Guidance | none | 6.87e-05 | 0.5035 | yes |
| Candor | direct | 5.73e-05 | 0.8526 | yes |
| Candor | evasive | 9.92e-06 | 0.1474 |  |
| Next EPS | up | 4.45e-05 | 0.5117 | yes |
| Next EPS | down | 4.25e-05 | 0.4883 |  |
| Action | BUY | 8.18e-06 | 0.4329 |  |
| Action | AVOID | 1.07e-05 | 0.5671 | yes |

## The two numbers that the scores use

- **p(Next EPS up)** = 4.45e-05 ÷ (4.45e-05 + 4.25e-05) = **0.5117**
- **p(BUY)** = 8.18e-06 ÷ (8.18e-06 + 1.07e-05) = **0.4329**. This is the stock score.

The published file has p(BUY) = 0.4329 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Demand | `<|im_start|>` 0.0013, `�` 1.47e-04, `�` 1.32e-04, `�` 9.06e-05, `�` 8.78e-05 |
| Pricing | `<|im_start|>` 0.0014, `�` 1.46e-04, `�` 1.37e-04, `�` 9.12e-05, `�` 8.70e-05 |
| Margins | `<|im_start|>` 0.0013, `�` 1.48e-04, `�` 1.39e-04, `�` 9.26e-05, `�` 8.98e-05 |
| Guidance | `<|im_start|>` 0.0014, `�` 1.47e-04, `�` 1.44e-04, `�` 9.62e-05, `�` 9.32e-05 |
| Candor | `<|im_start|>` 0.0015, `�` 1.46e-04, `�` 1.42e-04, `�` 9.46e-05, `�` 9.31e-05 |
| Next EPS | `<|im_start|>` 0.0014, `�` 1.47e-04, `�` 1.43e-04, `�` 9.65e-05, `�` 9.50e-05 |
| Action | `<|im_start|>` 0.0014, `�` 1.51e-04, `�` 1.47e-04, `�` 9.62e-05, `�` 9.18e-05 |
