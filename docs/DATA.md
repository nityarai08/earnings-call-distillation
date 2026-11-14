# Data

This page lists every data file that the pipeline reads or writes. For each file, it gives the source, the location, and the meaning of each column.

There are three kinds of files:

- **Downloaded**: public source tables. The small tables are in the repo. The script `reproduce/download_data.sh` downloads the large tables.
- **Built**: made by our scripts from the downloaded tables. In the repo when small and shareable.
- **Run outputs**: what the models produced during the runs. All in the repo, except rows from the sealed final test part.

## Sources

| Source | What we use | License / terms | In this repo? |
| --- | --- | --- | --- |
| [defeatbeta/yahoo-finance-data](https://huggingface.co/datasets/defeatbeta/yahoo-finance-data) on Hugging Face | Earnings-call transcripts, daily prices, financial statements, company profiles, dividends, splits, trailing EPS | ODC-BY 1.0. The dataset card says research and educational use | The small tables, yes. The transcripts, prices and statements, no: they are 2.9 GB, and the transcript text is not ours to share |
| [SEC company tickers](https://www.sec.gov/files/company_tickers.json) (copy inside the dataset above) | Company names, so they can be masked out of transcripts | US government data | Yes |
| Yahoo Finance via [yfinance](https://github.com/ranaroussi/yfinance) | SPY daily bars: the trading calendar and the market return | Yahoo terms | Yes, SPY only (`data/raw/daily_bars.parquet`) |
| [mlx-community/Qwen3.5-9B-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit), revision `8b2b98c00a6b4d291155e4890773ca8f769aee53` | The student model: a 4-bit MLX conversion of Qwen3.5-9B | See the model card | No. `reproduce/setup.sh` downloads it |
| [Claude Opus and Sonnet](https://docs.claude.com/en/docs/about-claude/models/overview) (`claude-opus-5-5`, `claude-sonnet-5-5`), Anthropic Batch API | Teacher labels and the analyst handbook | Anthropic terms | Their outputs, yes |
| [pi](https://github.com/earendil-works/pi) (`@earendil-works/pi-coding-agent@1.5.0` on npm) | The agent loop for the "Pi" setup | See its repo | Lockfile only. `npm ci` installs it |

The owners update the Hugging Face dataset every day. On 2026-11-13 they compacted its history. Thus nobody can download the 2026-11-06 snapshot that this run used. The file `data/raw/defeatbeta/spec.json` holds the dataset's own SHA-256 for each table in that snapshot. The script `reproduce/check_sources.py` shows whether your copies match.

## How calls are identified

Every earnings call has one integer id, `row`, used across all files.

- Rows below 200,000 are calls from 2016 to 2024 (built by `scripts/small_call_check.py`).
- Rows from 200,000 up are calls from 2025 and 2026 (built by `scripts/paper/build_data.py parse`).

**Decision timing.** A transcript gives the call date but no time. The model decides at the close of the first trading session after that date. The trade buys at the next open. It sells at the close 60 sessions after the decision.

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
| `data/paper/kb_outcomes.parquet` | 57,096 | Outcomes of all non-test calls. The library shows them as outcome cards, only after the trade closed |
| `data/paper/sealed/test_d.parquet`, `test_w.parquet` | 100, 800 | Outcomes of the two opened test parts |
| `data/paper/handbook.json` | 50 notes | General notes for reading earnings calls, written once by Claude Opus. They are part of the library |
| `data/cases/train.parquet` | 552 | `symbol` only: large stocks excluded from the universe because an earlier project already studied them |
| `reports/diagnostics/small_call_events.parquet` | 47,507 | 2016–2024 calls with outcomes, used to check that the rebuild matches the earlier study |
| `reports/diagnostics/small_call_read.csv` | 2,006 | `row` only: the 2023–2024 sample that became the `dev` and `student_check` sets |
| `configs/paper/freeze.json` | — | SHA-256 of `sets.parquet`, `outcomes.parquet` and each test part. The number of calls in each set. The row ids of each test part |

**`calls.parquet` / `sets.parquet` columns**

| Column | Meaning |
| --- | --- |
| `row` | Call id |
| `symbol` | Ticker (never shown to the model) |
| `call` | Call date |
| `fy`, `fq` | Fiscal year and quarter the call covers |
| `date` | Decision session (first close after the call) |
| `month` | `date` as `YYYY-MM`. The scoring script ranks calls within each month |
| `sector` | Sector from the company profile |
| `dollars` | Median daily dollar volume over the 60 sessions before the decision. The study uses only stocks from $1M to $50M, with a share price of at least $3 |
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
| `next_yoy` | Next quarter's EPS minus the same quarter a year earlier. Its sign is the answer to "Next EPS: up or down" |

### Teacher labels (`data/paper/teacher/`)

`labels_v2.jsonl` has one JSON object per line, one per (call, lens, model): 3,000 in all.

- 800 `sft` calls read by Opus with a `neutral` lens and again with a `skeptic` lens.
- 300 `student_check` calls read by Opus (`neutral`), and the same 300 read by Sonnet (`neutral`).
- 800 `test_w` calls read by Opus (`neutral`), with the same prompt and RAG block as the RAG setup (step `opus_test`, added 2026-11-14). They compare Opus with the small model in blog section 8. They were never used for training.

| Field | Meaning |
| --- | --- |
| `row`, `lens`, `model` | Which call, which reading instruction, which Claude model |
| `demand`, `pricing`, `margins`, `guidance`, `candor`, `next_eps` | The read lines, same values as the student's |
| `score` | Outlook for the stock over the next 60 sessions against its sector: 0 much worse, 50 in line, 100 much better. Turned into the soft BUY target for fine-tuning |
| `quotes` | Up to three short quotes from the call that drove the score |
| `tokens_in`, `tokens_out` | Tokens billed |

`labels.jsonl` is the first version from the mini run (260 lines, with a 1–5 `rating` instead of `score`).

### Real examples (`examples/`)

`examples/README.md` lists every experiment and test of the two runs, with one folder each. A folder holds the first 10 calls that the experiment used, in the fixed order of its call set. The test folders use the first 10 test calls of March 2026. `scripts/paper/examples.py` writes the folder.

| File | Content |
| --- | --- |
| `examples/prompts/<row>_<variant>.txt` | The exact text that a model read, chat template included. The variant names the prompt parts: `noblock` (no knowledge block), `rag`, `random`, `shuffled`, `pi`, `pi_rag_docs`, `nocall` (no call text), `holds` (HOLD or SELL), `noreaction`, `actiononly`, `habit`, `change`, and `opus_neutral` or `opus_skeptic` for Claude |
| `examples/<stage>/<step>/<experiment>/answers.csv` | For each call: the prompt file, the seven lines, p(Next EPS up), the probability of the first action, `p_mass`, and the real outcome (`excess_return`, `outcome_y`, `next_eps_change`) |
| `examples/week/sft_training/<variant>/targets.csv` | For each SFT training call: the prompt file and the target answer |
| `examples/week/rl_training/<reward>/rewards.csv` | For each RL training call: the reward inputs (`y`, `month_mean_y`, `z`, the reward of each action), the start model's answer and probabilities, the reward, and the RL target |
| `examples/week/rag/searches.csv` | For each example test call: the 12 query words, every passage that the search returned, its date and its outcome card |
| `examples/week/pi/agent_runs.csv` | For each example call: the searches that the Pi agent wrote and the documents that it kept |

The prompt files are the only place where transcript text appears in this repo.

### Full training data (`data/paper/training/`)

| File | Content |
| --- | --- |
| `sft_targets.csv` | One row for each SFT training call: 800 calls, plus 200 placebo calls. The columns are the actions, the target read lines, the real next EPS, Opus's two ratings and their mean. The last columns are the number of calls in the month and the target probability of the first action. The placebo rows have no ratings, because their targets came from other calls |
| `rl_rewards.csv` | One row for each RL training call: 400 calls, plus 200 calls of the shuffled control. The columns are the outcome y, the month mean, z, the daily volatility, and the reward of each action |

### Run outputs (`runs/paper/<stage>/`)

There is one folder for each stage, `mini` and `week`. The paper's numbers come from `week`.

| Path | What it holds |
| --- | --- |
| `results.csv` | Final test scores, one row per setup and baseline (columns in `blog/.../results/README.md`) |
| `findings.csv` | The four paired comparisons with 95% bootstrap ranges |
| `faithfulness.csv` | The forced-Demand test |
| `pick_sft.csv`, `pick_rl.csv`, `pick_block.csv` | Validation scores that picked the best SFT variant, RL reward and search setup. The picks are in `best_sft.txt`, `best_rl.txt` and `best_block.txt` |
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
| `p_mass` | For each line, the probability that the model gave to the allowed values, out of its whole vocabulary. The column keeps the lowest value of the seven lines. It is near 1 for the base model. It is near 0 for the trained models, because training compares the allowed values only with each other. Their stock scores are still valid. But in free text they would not keep the format |

File names:

- `raw`, `rag`, `pi`, `memory` (no call text): the setups
- `rag_random`, `rag_shuffled`: the RAG controls
- `pi_rag_docs`: the Pi loop with RAG's documents
- `prompt_*`: prompt styles
- `raw_flat`, `raw_long`: the position check
- `sft_*`, `rl_*`: trained variants. `sft_placebo` and `rl_shuffled` are controls
- `*_demand_up`, `*_demand_down`: the forced-Demand test
- `*_test`: the test set (Test-W)
- `*_check`: the 300 check calls that Opus also read

**`agent/*.jsonl` fields:** `row`, `code` (0 = finished), `seconds`, `calls` (tool calls), `tokens` (tokens generated), `queries` (the searches the model wrote), `docs` (knowledge-base document ids it kept). The published logs do not include the text of the documents, because it is transcript text. A new run of the agent makes it again.

## Not in this repo

| File | Why it is not here | How to make or download it |
| --- | --- | --- |
| `data/raw/transcripts/defeatbeta/transcripts.parquet` (2.3 GB), `data/raw/defeatbeta/stock_prices.parquet` (474 MB), `stock_statement.parquet` (121 MB) | Too large, and the transcripts are third-party text | `reproduce/download_data.sh` |
| `data/features/small_call_text.parquet`, `data/paper/text_new.parquet` | Masked transcript text | `reproduce/build_data.sh` |
| `data/paper/kb.sqlite` (1.1 GB) | The knowledge base: masked transcript passages | `reproduce/build_data.sh` |
| `data/paper/sealed/test_f.parquet` | Outcomes of the sealed final test part | Kept private until a full run scores it. `build_data.sh` builds it again on your computer |
| The other trained adapters and all optimizer states (about 3 GB) | Size | `reproduce/run_stage.sh` retrains them |
| Test-F rows in run outputs | Same reason as the sealed outcomes | Removed from every published file by `scripts/paper/export_repo.py` |
