# Sample output: sft_hard

SFT variant hard: Action trained on the real outcome

Call 200826. The model read the prompt in [examples/prompts/200826_rag_actiononly.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/200826_rag_actiononly.txt). The prompt ends with the label `Action:`.

## What the model wrote

```
Action: BUY
```

The model writes only these lines. It writes no probability. The script reads the probabilities from the model, as the next table shows.

## How the script read each line

Before each line value, the model gives a probability to every token in its vocabulary. The script takes the probability of each allowed value. Then it divides each value by the sum, so that the allowed values add up to 1. It keeps the most probable value, writes it, and goes to the next line.

| Line | Allowed value | Probability out of the whole vocabulary | Probability among the allowed values | Chosen |
| --- | --- | --- | --- | --- |
| Action | BUY | 0.1343 | 0.5467 | yes |
| Action | AVOID | 0.1113 | 0.4533 |  |

## The two numbers that the scores use

- **p(BUY)** = 0.1343 ÷ (0.1343 + 0.1113) = **0.5467**. This is the stock score.

The published file has p(BUY) = 0.5467 for this call. The values agree to the rounding of the published file.

## The five tokens that the model liked most at each line

These tokens show what the model wanted to write before the script limited it to the allowed values. A value with more than one token, for example " AVOID" or " evasive", counts by its first token (" A", " ev"). The sign � is a part of a character that takes more than one token.

| Line | Top five tokens (probability out of the whole vocabulary) |
| --- | --- |
| Action | `BUY` 0.1343, `A` 0.1113, `HOLD` 0.0165, `buy` 0.0137, `a` 0.0073 |
