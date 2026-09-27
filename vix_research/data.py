"""Validated adjusted daily data, optional Yahoo/FRED downloads, and fixtures."""
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
import hashlib
import json
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .core import ASSETS


@dataclass
class MarketData:
    opens: pd.DataFrame
    closes: pd.DataFrame
    rf_annual: pd.Series
    metadata: dict

    def validate(self):
        if self.closes.empty:
            raise ValueError("Empty market data")
        dates = self.closes.index
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError("Dates must be unique and increasing")
        if not dates.equals(self.opens.index):
            raise ValueError("Open/close calendars do not match")
        for ticker in (*ASSETS, "VIX"):
            if ticker not in self.closes:
                raise ValueError(f"Missing {ticker} closes")
            series = self.closes[ticker]
            first = series.first_valid_index()
            if first is None:
                raise ValueError(f"No data for {ticker}")
            values = series.loc[first:]
            if not np.isfinite(values.to_numpy()).all() or (values <= 0).any():
                raise ValueError(f"Missing/nonpositive {ticker} prices after inception; do not forward-fill")
            if ticker in ASSETS:
                op = self.opens[ticker].loc[first:]
                if not np.isfinite(op.to_numpy()).all() or (op <= 0).any():
                    raise ValueError(f"Missing/nonpositive {ticker} opens")
        if self.rf_annual.empty or self.rf_annual.index.has_duplicates:
            raise ValueError("A unique dated annual risk-free series is required")
        if not np.isfinite(self.rf_annual.to_numpy()).all() or (self.rf_annual <= -1).any():
            raise ValueError("Risk-free rates must be finite decimals greater than -1")
        return self

    def risk_free_returns(self):
        """Prior-session published annual yield, calendar-day accrual ACT/365.

        DGS3MO is a quoted Treasury yield, used as a cash-rate approximation,
        not as the realized return of a Treasury bond. No same-day rates.
        """
        dates = self.closes.index
        rates = self.rf_annual.sort_index()
        union = dates.union(rates.index).sort_values()
        aligned = rates.reindex(union).ffill().reindex(dates).shift(1)
        published = pd.Series(rates.index, index=rates.index).reindex(union).ffill()
        published = published.reindex(dates).shift(1)
        age = (pd.Series(dates, index=dates) - published).dt.days
        if ((age.iloc[1:] > 10) | aligned.iloc[1:].isna()).any():
            raise ValueError("Missing or stale (>10 days) risk-free observations")
        days = pd.Series(dates, index=dates).diff().dt.days.fillna(1)
        return ((1 + aligned) ** (days / 365.0) - 1).fillna(0.0)


def load_data(folder):
    folder = Path(folder)
    path = folder / "prices.csv"
    table = pd.read_csv(path, parse_dates=["date"])
    needed = {"date", "symbol", "adjusted_open", "adjusted_close"}
    if not needed.issubset(table.columns):
        raise ValueError(f"prices.csv requires {sorted(needed)}")
    if table.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate date/symbol rows")
    if table["date"].isna().any() or not table["date"].eq(table["date"].dt.normalize()).all():
        raise ValueError("Use session dates at midnight, without intraday timestamps")
    closes = table.pivot(index="date", columns="symbol", values="adjusted_close").sort_index()
    opens = table.pivot(index="date", columns="symbol", values="adjusted_open").sort_index()
    if "SPY" not in closes:
        raise ValueError("SPY is required as the trading calendar")
    dates = closes.index[closes["SPY"].notna()]
    closes, opens = closes.reindex(dates), opens.reindex(dates)
    rate_path = folder / "risk_free.csv"
    rates = pd.read_csv(rate_path, parse_dates=["date"])
    if not {"date", "annual_rate"}.issubset(rates.columns):
        raise ValueError("risk_free.csv requires date,annual_rate (decimal, not percent)")
    rf = rates.set_index("date")["annual_rate"].sort_index()
    manifest = folder / "manifest.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    metadata["prices_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata["risk_free_sha256"] = hashlib.sha256(rate_path.read_bytes()).hexdigest()
    metadata.setdefault("source", "user supplied adjusted CSV")
    metadata.setdefault("synthetic", False)
    return MarketData(opens.loc[:, ASSETS], closes.loc[:, (*ASSETS, "VIX")], rf, metadata).validate()


def save_data(data, folder):
    data.validate()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for symbol in (*ASSETS, "VIX"):
        close = data.closes[symbol].dropna()
        op = data.opens[symbol].reindex(close.index) if symbol in ASSETS else close
        records.append(pd.DataFrame({"date": close.index, "symbol": symbol,
                                     "adjusted_open": op.to_numpy(),
                                     "adjusted_close": close.to_numpy()}))
    pd.concat(records, ignore_index=True).to_csv(folder / "prices.csv", index=False)
    data.rf_annual.rename("annual_rate").rename_axis("date").to_csv(folder / "risk_free.csv")
    (folder / "manifest.json").write_text(json.dumps(data.metadata, indent=2), encoding="utf-8")


def download_data(folder, start="2006-01-01", end=None):
    """Yahoo adjusted OHLC plus FRED DGS3MO; no brokerage credentials needed."""
    import yfinance as yf

    cache = Path(folder) / ".yfinance_cache"
    cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache.resolve()))
    today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    end = end or today  # exclusive: never include the forming session
    if pd.Timestamp(end) > pd.Timestamp(today):
        raise ValueError("end must not include a future/forming New York session")
    closes, opens = {}, {}
    for symbol in (*ASSETS, "VIX"):
        ticker = "^VIX" if symbol == "VIX" else symbol
        print(f"Downloading {ticker} ...", flush=True)
        frame = yf.download(ticker, start=start, end=end, auto_adjust=True,
                            actions=False, progress=False, threads=False,
                            multi_level_index=False, timeout=30)
        if frame is None or frame.empty:
            raise RuntimeError(f"No Yahoo data for {ticker}; retry or import adjusted CSVs")
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
        closes[symbol], opens[symbol] = frame["Close"], frame["Open"]
    calendar = closes["SPY"].index
    close_frame = pd.DataFrame(closes).reindex(calendar)
    open_frame = pd.DataFrame(opens).reindex(calendar).loc[:, ASSETS]
    url = ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
           f"&cosd={start}&coed={end}")
    print("Downloading FRED DGS3MO cash-rate proxy ...", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "VIXResearch/1.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        raw = pd.read_csv(StringIO(response.read().decode("utf-8")), na_values=["."])
    rf = pd.Series(pd.to_numeric(raw["DGS3MO"], errors="coerce").to_numpy() / 100,
                   index=pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0]))).dropna()
    metadata = {"source": "Yahoo Finance via yfinance; FRED DGS3MO", "synthetic": False,
                "downloaded_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
                "requested_start": start, "end_exclusive": end,
                "price_adjustment": "Yahoo auto_adjust=True for both open and close",
                "risk_free": "Previous-session DGS3MO, decimal annual yield, ACT/365 approximation",
                "data_notice": "Research data can be revised. Check vendor terms and validate against LEAN."}
    data = MarketData(open_frame, close_frame, rf, metadata).validate()
    data.risk_free_returns()
    save_data(data, folder)
    return data


def synthetic_data(sessions=1600, seed=17):
    """Deterministic plumbing fixture. Never evidence of strategy performance."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", periods=sessions)
    fear = np.zeros(sessions)
    for start, length, level in ((350, 70, 24), (700, 50, 35), (1150, 90, 20)):
        fear[start:min(start + length, sessions)] = level
    vix = np.maximum(9, 15 + fear + rng.normal(0, 2, sessions))
    shock = rng.normal(0, 1, sessions)
    returns = np.column_stack([
        .00035 - fear * .000045 + shock * (.008 + fear * .00025),
        .00008 - shock * .0015 + rng.normal(0, .003, sessions),
        .00012 + rng.normal(0, .009, sessions),
        .00008 + rng.normal(0, .0001, sessions),
    ])
    closes = pd.DataFrame(np.array([200, 90, 120, 85]) * np.cumprod(1 + returns, axis=0),
                          index=dates, columns=ASSETS)
    previous = closes.shift(1).fillna(closes.iloc[0])
    gaps = rng.normal(0, [.003, .001, .003, .00003], size=(sessions, 4))
    opens = previous * (1 + gaps)
    closes["VIX"] = vix
    rf = pd.Series(.02 + .015 * np.sin(np.arange(sessions) / 250), index=dates)
    return MarketData(opens, closes, rf,
                      {"source": "synthetic fixture", "synthetic": True, "seed": seed}).validate()
