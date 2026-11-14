# Third-party data and software

**defeatbeta/yahoo-finance-data**
- Source: https://huggingface.co/datasets/defeatbeta/yahoo-finance-data
- License: Open Data Commons Attribution License (ODC-BY) 1.0.
- What's here: several small tables from its 2026-11-06 snapshot, in `data/raw/defeatbeta/`. Every file under `data/paper/`, `data/features/` and `reports/diagnostics/` is derived from that dataset.
- Not included: the earnings-call transcripts, except the example calls in `examples/prompts/`, whose full prompts are shown so that readers can see the real inputs. Short quotes in `data/paper/teacher/*.jsonl` were selected by Claude from those transcripts.

**SEC company tickers**
- Source: https://www.sec.gov/files/company_tickers.json, as included in the dataset above.

**SPY daily bars**
- Source: Yahoo Finance, fetched with yfinance (https://github.com/ranaroussi/yfinance).
- What's here: `data/raw/daily_bars.parquet`.

**Claude**
- Teacher labels and the analyst handbook were produced by Anthropic's Claude Opus and Claude Sonnet models through the Anthropic API.

**Qwen3.5-9B**
- Source: the 4-bit MLX conversion at https://huggingface.co/mlx-community/Qwen3.5-9B-4bit.
- The model is downloaded by `reproduce/setup.sh` and is not redistributed here.

**pi**
- Source: https://github.com/earendil-works/pi
- Installed from npm by `reproduce/setup.sh` and pinned in `tools/pi/package-lock.json`.
