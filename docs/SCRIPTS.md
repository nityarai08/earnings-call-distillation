# Scripts

This page explains what each script does, what it reads and writes, and the order they run in. Every command runs from the repo root with:

```bash
export PYTHONPATH=src:scripts
PY=".venv/bin/python -W ignore"
```

## The pipeline at a glance

```
download_data.sh ──► small_call_check.py build ──► build_data.py parse ─► market ─► sets ──► build_kb.py build ─► handbook ─► audit
 (source tables)      (2016–2024 call text)        (2025–2026 call text) (prices,    (fixed     (knowledge base: masked passages,
                                                                          universe,   sets,      facts, past-call cards)
                                                                          outcomes)   hashes)
                                                     │
                                                     ▼
                          run_step.py <stage>: runs configs/paper/commands.yaml step by step
   baselines ─► raw ─► rag ─► opus_check ─► opus_labels ─► sft ─► rl ─► pi ─► test ─► report
   (rules)     (9B)  (9B +   (Opus, Sonnet  (Opus labels  (fine- (RL)  (agent (sealed  (results.csv,
                      docs)   vs 9B)         for SFT)      tune)        loop)  test)   findings.csv)
```

The `reproduce/` scripts wrap these steps, and `reproduce/rerun_trained_reads.sh` reruns the trained models' test reads from the published adapters. See the [README](../README.md) for the three levels of reproduction.

## Building the data

| Script | Command | Reads | Writes |
| --- | --- | --- | --- |
| `scripts/small_call_check.py` | `build` | Transcripts, `stock_profile`, `data/cases/train.parquet` | `data/features/small_call_features.parquet` (word-list features), `small_call_text.parquet` (masked answers) for 2016–2024 calls |
| | `test` | The above, prices | `reports/diagnostics/small_call_events.parquet` and the word-list study (not needed to rebuild the paper data; the events file is in the repo) |
| `scripts/paper/build_data.py` | `parse` | Transcripts | `data/paper/text_new.parquet`, `feat_new.parquet`: the same for 2025–2026 calls (rows from 200,000) |
| | `market` | Features, prices, dividends, splits, profiles, EPS, SPY bars | `data/paper/calls.parquet` (what was knowable at the decision) and the outcomes, kept apart |
| | `sets` | `calls.parquet`, `reports/diagnostics/small_call_read.csv` | `sets.parquet`, `outcomes.parquet`, `kb_outcomes.parquet`, `sealed/test_{d,w,f}.parquet`, `configs/paper/freeze.json` |
| | `refresh_eps` | Today's EPS table (or the published one with `PAPER_PINNED_EPS=1`) | Fills next-quarter EPS for calls reported after the build. The `test` step runs it first |
| `scripts/paper/teacher.py` | `handbook` | — | `data/paper/handbook.json` (one Opus call, about $1.50; already in the repo) |
| `scripts/paper/build_kb.py` | `build` | Masked text, statements, `kb_outcomes.parquet` | `data/paper/kb.sqlite`: passages of about 220 words, company facts, and cards saying how past calls turned out |
| | `handbook` | `handbook.json` | Adds the handbook notes to the knowledge base |
| | `audit` | `kb.sqlite` | Checks that no passage is available before its date, and that no test call has a card |

**Masking.** Company names, tickers and years are replaced in every text the models see (`tagent.earnings_text.mask`).

**Dates in the knowledge base.** Every passage has an `available` date, and a search returns only what was available before the decision date (`scripts/paper/kb.py`).

## Running the experiments

| Script | What it does |
| --- | --- |
| `scripts/paper/make_commands.py` | Writes `configs/paper/commands.yaml`: the same steps for every stage, with that stage's sizes. Sizes are at the top of the file |
| `scripts/paper/run_step.py` | Runs one step or a whole stage from `commands.yaml`. Skips finished commands, retries a failed one once, then stops and marks the step failed. `run_step.py week` or `run_step.py week sft` |
| `scripts/llm_locked.py` | Runs a command while holding `logs/llm.lock`, so two 9B jobs never load at once on an 18 GB Mac |
| `scripts/paper/prompt.py` | The one prompt and seven-line output format shared by every setup, the teacher and training |
| `scripts/paper/blocks.py` | Builds the knowledge block a setup adds to the prompt (RAG documents, random or shuffled controls, or pi's picks), capped at 1,500 tokens |
| `scripts/paper/kb.py` | Searches the knowledge base (SQLite full-text search) with the date rule |
| `scripts/paper/read.py` | Every 9B read: one forward pass per call, saving the exact probability of each answer value. Options select the setup: `--block rag`, `--adapter`, `--no-text`, `--force Demand=up`, `--position long`, `--hidden` |
| `scripts/paper/agent_run.py` | Runs the pi agent once per call against a local model server and `kb_service.py`, inside a macOS sandbox with no network. `--forced` hands it RAG's documents instead (the control) |
| `scripts/paper/kb_service.py` | The local search service the pi tools call (`tools/pi/kb_tools.ts`) |
| `scripts/paper/teacher.py` | Sends calls to Claude through the Batch API (`submit`, `wait`, `fetch`). Skips any call already labelled, and refuses to go over the spending cap in `runs/paper/spend.csv` |
| `scripts/paper/train_soft.py` | LoRA fine-tuning (`sft`) and reinforcement learning (`rl`) on the 9B, fitted to an 18 GB Mac: the frozen layers run once, and only the top 8 layers get gradients. RL samples 4 reads per call and moves toward a reward-tilted target. Rewards: `business` (next EPS right), `trade` (60-day outcome), `both`, and `shuffled` (outcomes shuffled within each month) |
| `scripts/paper/score.py` | Scores reads the same way everywhere: IC within each month, accuracy, the BUY book after costs, and paired bootstrap ranges. The only script that opens a sealed test part, and only by name (`final --open test_w`) |
| `scripts/paper/interim.py`, `tracker.py` | Early development results and the progress dashboard while a stage runs (optional) |
| `scripts/paper/export_repo.py` | Builds this repo from the working project: copies code and data, trims three inputs, drops Test-F rows and retrieved transcript text, and stops if a key, private email or sealed file would leave |

## Stages

`configs/paper/steps.yaml` lists the stages; only the sizes differ between them.

| Stage | Test part | Development calls | SFT / RL calls | Machine time |
| --- | --- | --- | --- | --- |
| `mini` | Test-D, 100 calls | 100 | 80 / 16 per arm | about a day |
| `week` | Test-W, 800 calls | 500 | 800 / 400 per arm | about six days |
| `full` | Test-F, 1,757 calls | 3,000 | 3,000 / 800 per arm | about 25 days (not run yet) |

## Older modules the pipeline imports

Some helpers come from the project's earlier work and are included as they are:

- `scripts/small_call_check.py`, `call_qa_check.py`, `filing_change_check.py` and `raw_text_check.py`: price tables, the universe filter, the outcome definition and the word lists.
- `scripts/small_call_read.py`: company names for masking.
- `src/tagent/`: the trading calendar (`data.py`), dividend and split adjustment (`risk_agent.py`), the masking function (`earnings_text.py`), a Newey-West t-statistic (`evaluate.py`), and a memory-saving attention path for training (`train_lora.py`).

Some of these modules also hold code for earlier studies, which the paper does not run.

## Requirements

| To do | You need |
| --- | --- |
| Re-score the published outputs | macOS or Linux, Python 3.10+, `requirements-rescore.txt` |
| Rebuild the data | An Apple Silicon Mac set up by `reproduce/setup.sh`, and about 6 GB of disk |
| Rerun a stage | An Apple Silicon Mac with 18 GB of memory or more, Python 3.12, Node 22.19+, `requirements.txt`, and the 9B model (`reproduce/setup.sh`). An Anthropic API key only if you want fresh Opus labels |
