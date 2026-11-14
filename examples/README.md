# Worked examples

These are three real calls from the run: exactly what the models read, what they answered, what they were trained toward, and what happened next. Each prompt file is the full text the model saw, chat template included; it ends at `Demand:`, where the model starts answering.

## 1. A test call (`1_test_call/`)

This is the first of the 800 Test-W calls, by the set's fixed random order. Its outcome was sealed until the final scoring step.

| File | What it is |
| --- | --- |
| `prompt_memory.txt` | Memory only: the facts line, but no transcript and no knowledge |
| `prompt_raw.txt` | Raw: facts and management's answers |
| `prompt_rag.txt` | RAG: facts, the knowledge block from the fixed search, and the answers. The SFT and RL models read this same prompt |
| `prompt_pi.txt` | Pi: facts, the documents the model chose with its own searches, and the answers |
| `answers.csv` | What each of the six setups answered: the seven lines, p(Next EPS up) and p(BUY) |
| `outcome.json` | What happened: return, sector return, excess return, volatility, the outcome `y` and next quarter's EPS, step by step |

## 2. A fine-tuning call (`2_sft_training_call/`)

This is the first call the SFT model was trained on.

| File | What it is |
| --- | --- |
| `opus_labels.jsonl` | Claude Opus's two answers: neutral and skeptical, each with a 0–100 score and quotes |
| `target.json` | The training target: Opus's read lines, the real next-quarter EPS, and the BUY probability made from Opus's score |
| `training_text.txt` | The prompt followed by the target answer. The Action line is trained toward the probability in `target.json`, not toward a hard BUY |

## 3. A reinforcement-learning call (`3_rl_training_call/`)

This is the first RL call with a flat position and a clear outcome.

| File | What it is |
| --- | --- |
| `prompt.txt` | The prompt the RL model read |
| `reward.json` | The trade reward worked out step by step, the starting model's read and p(BUY), and the target RL moved it toward |

To regenerate these files (needs the knowledge base from `reproduce/build_data.sh`):

```bash
PYTHONPATH=src:scripts .venv/bin/python -W ignore scripts/paper/examples.py --model
```

The blog post explains every number: [`blog/2026-11-small-model-earnings-calls/README.md`](../blog/2026-11-small-model-earnings-calls/README.md).
