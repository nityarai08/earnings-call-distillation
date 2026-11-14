# Data

This page lists every data file the pipeline reads or writes: where it comes from, whether it is in this repo, and what its columns mean.

There are three kinds of files:

- **Downloaded**: public source tables. Small ones are in the repo; large ones are fetched by `reproduce/download_data.sh`.
- **Built**: made by our scripts from the downloaded tables. In the repo when small and shareable.
- **Run outputs**: what the models produced during the runs. All in the repo, except rows from the sealed final test part.

## Sources

| Source | What we use | License / terms | In this repo? |
| --- | --- | --- | --- |
| [defeatbeta/yahoo-finance-data](https://huggingface.co/datasets/defeatbeta/yahoo-finance-data) on Hugging Face | Earnings-call transcripts, daily prices, financial statements, company profiles, dividends, splits, trailing EPS | ODC-BY 1.0; the card says research and educational use | Small tables yes; transcripts, prices and statements no (2.9 GB, and the transcript text is not ours to redistribute) |
| [SEC company tickers](https://www.sec.gov/files/company_tickers.json) (copy inside the dataset above) | Company names, so they can be masked out of transcripts | US government data | Yes |
| Yahoo Finance via [yfinance](https://github.com/ranaroussi/yfinance) | SPY daily bars: the trading calendar and the market return | Yahoo terms | Yes, SPY only (`data/raw/daily_bars.parquet`) |
| [mlx-community/Qwen3.5-9B-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit), revision `8b2b98c00a6b4d291155e4890773ca8f769aee53` | The student model: a 4-bit MLX conversion of Qwen3.5-9B | See the model card | No; `reproduce/setup.sh` downloads it |
| [Claude Opus and Sonnet](https://docs.claude.com/en/docs/about-claude/models/overview) (`claude-opus-5-5`, `claude-sonnet-5-5`), Anthropic Batch API | Teacher labels and the analyst handbook | Anthropic terms | Their outputs, yes |
| [pi](https://github.com/earendil-works/pi) (`@earendil-works/pi-coding-agent@1.5.0` on npm) | The agent loop for the "Pi" setup | See its repo | Lockfile only; `npm ci` installs it |

The Hugging Face dataset is updated daily, and its history was compacted on 2026-11-13, so the exact 2026-11-06 snapshot this run used can no longer be downloaded. `data/raw/defeatbeta/spec.json` holds the dataset's own SHA-256 for every table in that snapshot; `reproduce/check_sources.py` tells you whether your copies still match.

## How calls are identified

Every earnings call has one integer id, `row`, used across all files.

- Rows below 200,000 are calls from 2016 to 2024 (built by `scripts/small_call_check.py`).
- Rows from 200,000 up are calls from 2025 and 2026 (built by `scripts/paper/build_data.py parse`).

**Decision timing.** A transcript gives the call date but no time. The model decides at the close of the first trading session after that date, buys at the next open, and sells at the close 60 sessions after the decision.

## The sets

`build_data.py sets` assigns each call to one set before any experiment runs, and records the assignment and its hashes in `configs/paper/freeze.json`.

| Set | Calls | Dates | Used for |
| --- | --- | --- | --- |
| `sft` | 3,000 | 2021-01 to 2024-10, even quarters | Fine-tuning on Opus labels (the week run used 800) |
| `sft_val` | 300 | same | Held-out loss during fine-tuning |
| `rl` | 800 | 2021-01 to 2024-10, odd quarters | Reinforcement learning (the week run used 400) |
| `dev` | 1,002 | 2023 to 2024 | Prompt checks in the mini run |
| `student_check` | 1,000 | 2023 to 2024 | Comparing each setup's read with Opus's read of the same call |
| `validation` | 3,000 | 2025-01 to 2025-11 | Tuning and every development comparison in the week run |
| `test_d` | 100 | 2026-03 to 2026-08 | Final test of the mini run (opened) |
| `test_w` | 800 | 2026-03 to 2026-08 | Final test of the week run (opened) |
| `test_f` | 1,757 | 2026-03 to 2026-08 | Final test of a full run. **Sealed: its outcomes are not in this repo.** |

The three test parts are split at random within each month. Within a set, `rank` gives a fixed random order, so a stage that uses n calls takes `rank < n`. That makes the mini run's calls a subset of the week's.

## Files in this repo

### Downloaded (`data/raw/`)

| File | Rows | Columns |
| --- | --- | --- |
| `defeatbeta/spec.json` | — | The dataset's own SHA-256 and update time for each table, as of 2026-11-06 |
| `defeatbeta/company_tickers.json` | — | SEC ticker → company name, used to mask names |
| `defeatbeta/stock_profile.parquet` | 11,593 | `symbol`, `sector`, `industry`, address and summary fields, `report_date` |
| `defeatbeta/stock_dividend_events.parquet` | 308,721 | `symbol`, `report_date`, `amount` |
| `defeatbeta/stock_split_events.parquet` | 10,188 | `symbol`, `report_date`, `split_factor` |
| `defeatbeta/stock_tailing_eps.parquet` | 512,276 | `symbol`, `report_date` (quarter end), `eps`, `tailing_eps`, `update_time`. This is the copy refreshed on 2026-11-12, which labels quarters reported after the build |
| `daily_bars.parquet` | 3,227 | SPY only: `date`, `open`, `high`, `low`, `close`, `adj_close`, `volume`, `feed`, `received_at_utc` |

### Built inputs

| File | Rows | What it is |
| --- | --- | --- |
| `data/features/small_call_features.parquet` | 104,019 | Word-list features of 2016–2024 calls (columns as in `calls.parquet` below, plus `prepared_words`) |
| `data/paper/feat_new.parquet` | 26,685 | The same features for 2025–2026 calls |
| `data/paper/calls.parquet` | 59,753 | Every call in the universe, with only what was knowable at the decision (columns below) |
| `data/paper/sets.parquet` | 11,759 | The calls placed in a set: the `calls.parquet` columns plus `set` and `rank` |
| `data/paper/outcomes.parquet` | 9,102 | Outcomes of the non-test sets (columns below) |
| `data/paper/kb_outcomes.parquet` | 57,096 | Outcomes of all non-test calls; the knowledge base shows these as "how a past call turned out" cards, only after the outcome closed |
| `data/paper/sealed/test_d.parquet`, `test_w.parquet` | 100, 800 | Outcomes of the two opened test parts |
| `data/paper/handbook.json` | 50 notes | General notes for reading earnings calls, written once by Claude Opus; part of the knowledge base |
| `data/cases/train.parquet` | 552 | `symbol` only: large stocks excluded from the universe because an earlier project already studied them |
| `reports/diagnostics/small_call_events.parquet` | 47,507 | 2016–2024 calls with outcomes, used to check that the rebuild matches the earlier study |
| `reports/diagnostics/small_call_read.csv` | 2,006 | `row` only: the 2023–2024 sample that became the `dev` and `student_check` sets |
| `configs/paper/freeze.json` | — | SHA-256 of `sets.parquet`, `outcomes.parquet` and each test part; calls per set; the row ids of each test part |

**`calls.parquet` / `sets.parquet` columns**

| Column | Meaning |
| --- | --- |
| `row` | Call id |
| `symbol` | Ticker (never shown to the model) |
| `call` | Call date |
| `fy`, `fq` | Fiscal year and quarter the call covers |
| `date` | Decision session (first close after the call) |
| `month` | `date` as `YYYY-MM`; IC is computed within each month |
| `sector` | Sector from the company profile |
| `dollars` | Median daily dollar volume over the 60 sessions before the decision; the universe is $1M–$50M, with a share price of at least $3 |
| `past` | Return over the 60 sessions before the decision |
| `risk` | Standard deviation of daily returns over those 60 sessions |
| `reaction` | Price move from before the call to the decision close, in units of daily volatility × √2 |
| `eps_yoy` | Latest reported quarterly EPS minus the same quarter a year earlier |
| `eps_change` | `eps_yoy` divided by the share price |
| `answer_tone`, `question_tone`, `prepared_tone` | Word-list tone of management's answers, analysts' questions and the prepared remarks |
| `answer_words`, `question_words` | Word counts (calls need ≥300 answer words and ≥100 question words) |
| `dodges` | Count of evasive phrases in the answers |
| `askers` | Number of analysts who asked questions |
| `set`, `rank` | (sets only) The set, and the call's fixed random order within it |

**Outcome columns** (`outcomes.parquet`, `kb_outcomes.parquet`, `sealed/*.parquet`)

| Column | Meaning |
| --- | --- |
| `row`, `symbol` | Call id and ticker |
| `date`, `exit` | Decision session and exit session (60 sessions later) |
| `ret` | Return from the next open to the exit close, dividends included |
| `excess` | `ret` minus the average return of the same sector's stocks |
| `y` | `excess` divided by (daily volatility × √60): the stock outcome every IC uses |
| `next_yoy` | Next quarter's EPS minus the same quarter a year earlier; its sign is what "Next EPS: up or down" predicts |

### Teacher labels (`data/paper/teacher/`)

`labels_v2.jsonl` has one JSON object per line, one per (call, lens, model): 2,200 in all.

- 800 `sft` calls read by Opus with a `neutral` lens and again with a `skeptic` lens.
- 300 `student_check` calls read by Opus (`neutral`), and the same 300 read by Sonnet (`neutral`).

| Field | Meaning |
| --- | --- |
| `row`, `lens`, `model` | Which call, which reading instruction, which Claude model |
| `demand`, `pricing`, `margins`, `guidance`, `candor`, `next_eps` | The read lines, same values as the student's |
| `score` | Outlook for the stock over the next 60 sessions against its sector: 0 much worse, 50 in line, 100 much better. Turned into the soft BUY target for fine-tuning |
| `quotes` | Up to three short quotes from the call that drove the score |
| `tokens_in`, `tokens_out` | Tokens billed |

`labels.jsonl` is the first version from the mini run (260 lines, with a 1–5 `rating` instead of `score`).

### Worked examples (`examples/`)

These are three real calls with everything the models saw and wrote, written by `scripts/paper/examples.py`. They are the only place where full transcript text appears in this repo.

| Folder | Files |
| --- | --- |
| `1_test_call/` (Test-W row 218058) | `prompt_memory.txt`, `prompt_raw.txt`, `prompt_rag.txt`, `prompt_pi.txt`: the exact text each setup read, chat template included. `answers.csv`: every setup's seven lines and probabilities. `outcome.json`: the return, sector return, volatility, `y` and next EPS, step by step |
| `2_sft_training_call/` (SFT row 60332) | `opus_labels.jsonl`: Opus's neutral and skeptical answers. `target.json`: the training target and how the BUY probability was derived. `training_text.txt`: the prompt followed by the target answer |
| `3_rl_training_call/` (RL row 4412) | `prompt.txt`, and `reward.json`: the trade-reward arithmetic, the starting model's read and the RL target |

### Run outputs (`runs/paper/<stage>/`)

There is one folder per stage, `mini` and `week`; the paper's numbers come from `week`.

| Path | What it holds |
| --- | --- |
| `results.csv` | Final test scores, one row per setup and baseline (columns in `blog/.../results/README.md`) |
| `findings.csv` | The four paired comparisons with 95% bootstrap ranges |
| `faithfulness.csv` | The forced-Demand test |
| `pick_sft.csv`, `pick_rl.csv`, `pick_block.csv` | Development scores used to pick the best SFT arm, RL arm and retrieval setup; the picks are in `best_sft.txt`, `best_rl.txt`, `best_block.txt` |
| `memo_memo.json` | Memorization check: how exactly the model repeats labels it was trained on |
| `reads/<arm>.csv` | One row per call for every model setup and control (below) |
| `reads/base_<rule>.csv`, `base_<rule>_test.csv` | Scores of the four simple rules on the development and test calls |
| `reads/probe_hidden_*.npz` | The 9B's last hidden state for the probe arm: `row` (n) and `h` (n × 4096), 100 calls per file |
| `agent/pi.jsonl`, `pi_rag_docs.jsonl`, `pi_test.jsonl` | One line per pi agent run (below) |
| `train/<arm>.csv` | Training curves for every SFT and RL arm, including the two runs that diverged at a learning rate of 1e-4 |
| `<step>/status.json`, `commands_done.json`, `DONE.json`, `run.log` | Runner bookkeeping and logs for each step |
| `opus_check/batches.jsonl`, `opus_labels/batches.jsonl` | Anthropic batch ids and sizes |

`runs/paper/spend.csv` is the API spending ledger: `at`, `dollars`, `what`.

### Trained adapters (`adapters/paper/week/`)

These are the LoRA weights of the two trained models scored on Test-W. Each folder holds `adapters.safetensors` (43 MB) and `adapter_config.json` (LoRA rank 16, scale 20, top 8 layers). They load with `mlx_lm.load("mlx-community/Qwen3.5-9B-4bit", adapter_path=...)`, which is what `read.py --adapter` does.

| Folder | Model |
| --- | --- |
| `sft_grown` | SFT: fine-tuned on Opus's labels for 800 calls (`read_eps` on 400, then 400 more at a lower learning rate) |
| `trade` | RL: 100 updates with the trade reward over 400 calls, starting from `sft_grown` (picked in `best_rl.txt`) |

Rerunning the first test calls with these files gives the published probabilities exactly (`reproduce/rerun_trained_reads.sh`).

**`reads/<arm>.csv` columns**

| Column | Meaning |
| --- | --- |
| `row` | Call id |
| `position` | `flat` (choose BUY or AVOID) or `long` (choose HOLD or SELL) |
| `tokens`, `seconds` | Prompt length and time taken |
| `demand`, `pricing`, `margins`, `guidance`, `candor` | The model's most likely value for each read line |
| `p_eps_up` | Probability of "Next EPS: up" (empty for arms that do not write that line) |
| `p_pos` | Probability of the positive action (BUY when flat, HOLD when long): the stock score |
| `p_mass` | The lowest, over the seven lines, of the probability the model put on that line's allowed values out of its whole vocabulary. Near 1 for the base model. Near 0 for the trained models, because training compares the allowed values only with each other: their scores are still well defined, but left to write freely they would not keep the format |

Arm names: `raw`, `rag`, `pi`, `memory` (no transcript), `rag_random` and `rag_shuffled` (retrieval controls), `pi_rag_docs` (pi loop with RAG's documents), `prompt_*` (prompt styles), `raw_flat` and `raw_long` (position check), `sft_*` and `rl_*` (trained arms), `sft_placebo` and `rl_shuffled` (controls), `*_demand_up` and `*_demand_down` (forced Demand), `*_test` (Test-W), `*_check` (the 300 `student_check` calls Opus also read).

**`agent/*.jsonl` fields:** `row`, `code` (0 = finished), `seconds`, `calls` (tool calls), `tokens` (tokens generated), `queries` (the searches the model wrote), `docs` (knowledge-base document ids it kept). The retrieved text itself is left out of the published logs because it is transcript text; rerunning the agent regenerates it.

## Not in this repo

| File | Why | How to get it |
| --- | --- | --- |
| `data/raw/transcripts/defeatbeta/transcripts.parquet` (2.3 GB), `data/raw/defeatbeta/stock_prices.parquet` (474 MB), `stock_statement.parquet` (121 MB) | Too large, and the transcripts are third-party text | `reproduce/download_data.sh` |
| `data/features/small_call_text.parquet`, `data/paper/text_new.parquet` | Masked transcript text | `reproduce/build_data.sh` |
| `data/paper/kb.sqlite` (1.1 GB) | The knowledge base: masked transcript passages | `reproduce/build_data.sh` |
| `data/paper/sealed/test_f.parquet` | Outcomes of the sealed final test part | Withheld until a full run scores it; `build_data.sh` rebuilds it locally |
| The other trained adapters and all optimizer states (about 3 GB) | Size | `reproduce/run_stage.sh` retrains them |
| Test-F rows in run outputs | Same reason as the sealed outcomes | Removed from every published file by `scripts/paper/export_repo.py` |
