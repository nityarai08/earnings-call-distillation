"""Daily bar ingestion and point-in-time access.

Raw bars are stored once as immutable Parquet (one file per download run) and
every downstream consumer reads through `load_bars(asof=...)`, which drops any
row dated after the decision time.
"""
from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
BARS_PATH = RAW_DIR / "daily_bars.parquet"


def load_universe() -> dict:
    return yaml.safe_load((ROOT / "configs" / "universe.yaml").read_text())


UNIVERSE_DIR = RAW_DIR / "universe"


def all_symbols() -> list[str]:
    u = load_universe()
    return u["watchlist_v1"] + u["context"]


def sp500_membership() -> pd.DataFrame:
    """Point-in-time S&P 500 membership intervals (fja05680/sp500, MIT).
    end_date NaT = still a member. Tickers use Yahoo's '-' class separator."""
    m = pd.read_csv(UNIVERSE_DIR / "sp500_ticker_start_end.csv",
                    parse_dates=["start_date", "end_date"])
    m["ticker"] = m["ticker"].str.replace(".", "-", regex=False)
    return m


def sp500_members_between(start: str, end: str) -> list[str]:
    m = sp500_membership()
    w = (m.start_date <= pd.Timestamp(end)) & (m.end_date.isna() | (m.end_date >= pd.Timestamp(start)))
    return sorted(m[w].ticker.unique())


def is_sp500_member(symbols: pd.Series, dates: pd.Series) -> pd.Series:
    """Vectorized point-in-time membership test for (symbol, date) pairs."""
    m = sp500_membership()
    df = pd.DataFrame({"ticker": symbols.values, "date": dates.values, "i": range(len(symbols))})
    j = df.merge(m, on="ticker", how="left")
    ok = (j.start_date <= j.date) & (j.end_date.isna() | (j.end_date > j.date))
    return pd.Series(ok.groupby(j.i).any().reindex(range(len(symbols)), fill_value=False).values,
                     index=symbols.index)


def sp100_test_universe() -> list[str]:
    # Snapshot taken 2026-11-01; membership changes during the test window are
    # minor but are a documented caveat (not strictly point-in-time).
    return pd.read_csv(UNIVERSE_DIR / "sp100_snapshot_20261101.csv").symbol.tolist()


def gics_sectors() -> dict[str, str]:
    return pd.read_csv(UNIVERSE_DIR / "gics_sectors.csv").set_index("symbol").gics_sector.to_dict()


def download_daily(start: str = "2014-01-01", end: str | None = None,
                   symbols: list[str] | None = None, chunk: int = 100) -> pd.DataFrame:
    """Download split-adjusted daily OHLCV from Yahoo.

    Long format: date, symbol, open, high, low, close, adj_close, volume.
    `close` is split-adjusted only; `adj_close` is split+dividend adjusted.
    Delisted tickers Yahoo no longer serves are silently missing (survivorship
    caveat for the train split; coverage is reported by dataset.py).
    """
    syms = symbols or all_symbols()
    parts = [_download_chunk(syms[i:i + chunk], start, end) for i in range(0, len(syms), chunk)]
    return pd.concat([p for p in parts if not p.empty], ignore_index=True)


def _download_chunk(syms: list[str], start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    warnings.filterwarnings("ignore", category=Warning, module="yfinance")
    raw = yf.download(syms, start=start, end=end, auto_adjust=False,
                      progress=False, group_by="column", threads=True)
    if raw.empty:
        return pd.DataFrame()
    frames = []
    for s in syms:
        try:
            df = pd.DataFrame({
                "open": raw["Open"][s], "high": raw["High"][s], "low": raw["Low"][s],
                "close": raw["Close"][s], "adj_close": raw["Adj Close"][s],
                "volume": raw["Volume"][s],
            })
        except KeyError:
            continue
        df = df.dropna(subset=["close"])
        df["symbol"] = s
        frames.append(df)
    out = pd.concat(frames).reset_index().rename(columns={"Date": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out["feed"] = "yahoo_daily"
    out["received_at_utc"] = pd.Timestamp.utcnow().tz_localize(None)
    return out[["date", "symbol", "open", "high", "low", "close", "adj_close",
                "volume", "feed", "received_at_utc"]]


def save_bars(df: pd.DataFrame) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # immutable snapshot plus a "current" pointer file
    snap = RAW_DIR / f"daily_bars_{date.today():%Y%m%d}.parquet"
    df.to_parquet(snap, index=False)
    df.to_parquet(BARS_PATH, index=False)
    return snap


def load_bars(asof: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """All bars with date <= asof (a daily bar is observable after its close)."""
    df = pd.read_parquet(BARS_PATH)
    if asof is not None:
        df = df[df["date"] <= pd.Timestamp(asof)]
    return df


def wide(df: pd.DataFrame, field: str) -> pd.DataFrame:
    return df.pivot(index="date", columns="symbol", values=field).sort_index()


def broad_symbols() -> list[str]:
    """Everything the broad train/val/test splits need."""
    return sorted(set(all_symbols()) | set(sp500_members_between("2015-01-01", "2026-12-31"))
                  | set(sp100_test_universe()))


if __name__ == "__main__":
    import sys
    bars = download_daily(symbols=broad_symbols() if "--broad" in sys.argv else None)
    p = save_bars(bars)
    print(f"saved {len(bars):,} rows, {bars.symbol.nunique()} symbols, "
          f"{bars.date.min():%Y-%m-%d}..{bars.date.max():%Y-%m-%d} -> {p}")
