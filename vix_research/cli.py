"""Command-line entry point: python -m vix_research --help."""
import argparse
from pathlib import Path
import json

import pandas as pd

from .core import StrategyConfig
from .data import load_data, save_data, download_data, synthetic_data
from .backtest import ExecutionConfig, run_backtest, compare, comparison_variants
from .analytics import (comparison_table, cost_stress, stress_variants, walk_forward,
                        performance, yearly_metrics, monthly_returns, block_bootstrap)
from .report import write_report, save_json


def build_quantconnect(destination=None):
    base = Path(__file__).parent
    destination = Path(destination) if destination else base / "VIX Fear-Regime Multi-Asset Strategy.py"
    core = (base / "core.py").read_text(encoding="utf-8")
    adapter = (base / "qc_adapter.py").read_text(encoding="utf-8")
    adapter = "\n".join(line for line in adapter.splitlines()
                        if not line.startswith("from vix_research.core import"))
    text = ("# GENERATED from vix_research/core.py and qc_adapter.py.\n"
            "# Rebuild with: python -m vix_research build-qc\n"
            "# Gold is a commodity allocation; IEF and BIL are Treasury allocations.\n"
            "# Local research and this adapter share signals, but differ in execution.\n\n"
            + core + "\n\n" + adapter + "\n")
    compile(text, str(destination), "exec")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description="VIX multi-asset strategy research toolkit")
    sub = parser.add_subparsers(dest="command", required=True)
    literature = sub.add_parser("literature", help="Frozen literature study, chronological tests and attribution")
    literature.add_argument("--data", default="vix_research/data/market")
    literature.add_argument("--output", default="vix_research/outputs/literature")
    literature.add_argument("--stage", choices=("freeze", "develop", "test", "report", "all"), default="all")
    fetch = sub.add_parser("fetch", help="Download adjusted Yahoo prices and FRED cash rates")
    fetch.add_argument("--output", default="vix_research/data/market")
    fetch.add_argument("--start", default="2006-01-01")
    fetch.add_argument("--end", help="Exclusive end date; defaults to today in New York")
    demo = sub.add_parser("demo", help="Generate a clearly labelled synthetic end-to-end demonstration")
    demo.add_argument("--output", default="vix_research/outputs/demo")
    demo.add_argument("--sessions", type=int, default=1600)
    demo.add_argument("--seed", type=int, default=17)
    for command in ("backtest", "compare", "robustness", "walk-forward"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--data", default="vix_research/data/market")
        cmd.add_argument("--output", default=f"vix_research/outputs/{command}")
        cmd.add_argument("--config", help="JSON StrategyConfig overrides")
        cmd.add_argument("--start", help="First evaluation date; earlier input data is warm-up")
        cmd.add_argument("--end", help="Last evaluation date, inclusive")
        cmd.add_argument("--commission-bps", type=float, default=1)
        cmd.add_argument("--slippage-bps", type=float, default=5)
        cmd.add_argument("--cash", type=float, default=100_000)
        if command == "walk-forward":
            cmd.add_argument("--train-years", type=int, default=5)
            cmd.add_argument("--test-years", type=int, default=1)
    build = sub.add_parser("build-qc", help="Generate a standalone QuantConnect Python strategy")
    build.add_argument("--output")
    analyse = sub.add_parser("analyse", help="Analyse an exported daily.csv (returns and risk-free returns required)")
    analyse.add_argument("--daily", required=True)
    analyse.add_argument("--output", default="vix_research/outputs/analysis")
    args = parser.parse_args(argv)

    if args.command == "literature":
        from .literature_study import run
        run(args.output, args.data, args.stage)
        return

    if args.command == "build-qc":
        print(build_quantconnect(args.output))
        return
    if args.command == "fetch":
        data = download_data(args.output, args.start, args.end)
        print(f"Saved {len(data.closes)} sessions to {args.output}")
        return
    if args.command == "demo":
        data = synthetic_data(args.sessions, args.seed)
        save_data(data, Path(args.output) / "synthetic_inputs")
        # Reload to capture the same input hashes as real-data experiments.
        data = load_data(Path(args.output) / "synthetic_inputs")
        results = compare(data)
        print(write_report(results, args.output))
        print("SYNTHETIC DEMO ONLY. These returns provide no evidence of investment performance.")
        return
    if args.command == "analyse":
        daily = pd.read_csv(args.daily, parse_dates=["date"]).set_index("date")
        required = {"return", "risk_free_return", "turnover", "traded", "cost"}
        if not required.issubset(daily.columns):
            raise ValueError(f"Daily CSV requires {sorted(required)} plus date")
        if daily.index.has_duplicates or not daily.index.is_monotonic_increasing:
            raise ValueError("Dates must be sorted and unique")
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        save_json(output / "performance.json", performance(daily))
        yearly_metrics(daily).to_csv(output / "yearly_metrics.csv")
        monthly_returns(daily).to_csv(output / "monthly_returns.csv")
        block_bootstrap(daily).to_csv(output / "bootstrap_intervals.csv")
        print(json.dumps(performance(daily), indent=2))
        return

    overrides = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    config = StrategyConfig(**overrides)
    execution = ExecutionConfig(args.cash, args.commission_bps, args.slippage_bps)
    data = load_data(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "walk-forward":
        folds = walk_forward(data, config, execution, args.start, args.end,
                             args.train_years, args.test_years)
        folds.to_csv(output / "folds.csv", index=False)
        save_json(output / "manifest.json", {"config": config.to_dict(), "execution": execution.__dict__,
                  "data": data.metadata, "train_years": args.train_years, "test_years": args.test_years,
                  "method": "Training Sharpe selects among four presets. Independent flat-start test folds with terminal liquidation costs; not a continuous NAV."})
        print(folds[["test_start", "test_end", "selected", "test_sharpe", "fixed_revised_test_sharpe"]].to_string(index=False))
        return
    if args.command == "backtest":
        results = {"revised": run_backtest(data, config, execution, args.start, args.end)}
    elif args.command == "compare":
        results = compare(data, comparison_variants(config), execution, args.start, args.end)
    else:
        results = compare(data, stress_variants(config), execution, args.start, args.end)
        # Match cost stress dates to the common sensitivity comparison period.
        first = next(iter(results.values())).daily.index[0]
        cost_stress(data, config, execution, first, args.end).to_csv(output / "cost_stress.csv", index=False)
    print(comparison_table(results)[["cagr", "annual_vol", "sharpe", "max_drawdown"]].round(3).to_string())
    print(write_report(results, output, data=data))
