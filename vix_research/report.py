"""Standalone interactive HTML, publication-friendly PNG, and audit exports."""
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os

import numpy as np
import pandas as pd

from .analytics import (performance, comparison_table, drawdown, monthly_returns,
                        yearly_metrics, regime_attribution, rolling_metrics, block_bootstrap)
from .core import ASSETS


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (datetime, pd.Timestamp, Path)):
        return str(value)
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def format_comparison(table):
    display = table[["cagr", "annual_vol", "sharpe", "sortino", "max_drawdown", "calmar",
                     "gross_turnover_per_year", "total_cost_dollars"]].copy()
    for column in ("cagr", "annual_vol", "max_drawdown"):
        display[column] = display[column].map(lambda x: f"{x:.1%}")
    for column in ("sharpe", "sortino", "calmar", "gross_turnover_per_year"):
        display[column] = display[column].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "—")
    display["total_cost_dollars"] = display["total_cost_dollars"].map(lambda x: f"${x:,.0f}")
    display.columns = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max drawdown", "Calmar",
                       "Gross turnover / year", "Costs"]
    return display


def write_report(results, folder, data=None, bootstrap_samples=300):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((folder / ".matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    primary = results.get("revised", next(iter(results.values())))
    table = comparison_table(results)
    table.to_csv(folder / "comparison.csv")
    source_hash = {}
    for file in Path(__file__).parent.glob("*.py"):
        source_hash[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    versions = {}
    for package in ("numpy", "pandas", "matplotlib", "plotly", "yfinance"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    for name, result in results.items():
        dest = folder / name
        dest.mkdir(exist_ok=True)
        for label, frame in (("daily", result.daily), ("weights", result.weights),
                             ("trades", result.trades), ("signals", result.signals)):
            frame.to_csv(dest / f"{label}.csv", index=label != "trades")
        yearly_metrics(result.daily).to_csv(dest / "yearly_metrics.csv")
        monthly_returns(result.daily).to_csv(dest / "monthly_returns.csv")
        regime_attribution(result.daily).to_csv(dest / "regime_attribution.csv", index=False)
        rolling_metrics(result.daily).to_csv(dest / "rolling_metrics.csv")
        save_json(dest / "manifest.json", {"name": name, "config": result.config,
                  "execution": result.execution, "metadata": result.metadata,
                  "metrics": performance(result.daily), "versions": versions,
                  "source_sha256": source_hash, "created_utc": datetime.now(timezone.utc).isoformat()})
    ci = block_bootstrap(primary.daily, samples=bootstrap_samples,
                         block=min(20, len(primary.daily)))
    ci.to_csv(folder / "bootstrap_intervals.csv")
    if data is not None:
        asset_returns = data.closes.loc[primary.daily.index, list(ASSETS)].pct_change(fill_method=None)
        asset_returns.corr().to_csv(folder / "asset_correlations.csv")

    colours = ["#147d92", "#bc6c25", "#6f58a5", "#4a7c59", "#c8555c", "#64748b"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.045,
                        row_heights=[.36, .19, .20, .25],
                        subplot_titles=("Growth of $100,000 · after costs", "Drawdown",
                                        "Prior-session VIX used for decisions", "Actual closing allocations"))
    for i, (name, result) in enumerate(results.items()):
        visible = True if name in (primary.name, "spy_buy_hold", "spy_bil_no_vix", "static_vol_target") else "legendonly"
        growth = (1 + result.daily["return"]).cumprod() * 100_000
        fig.add_trace(go.Scatter(x=result.daily.index, y=growth, name=name,
                                 legendgroup=name, visible=visible,
                                 line=dict(color=colours[i % len(colours)], width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=result.daily.index, y=drawdown(result.daily["return"]),
                                 name=name, legendgroup=name, visible=visible, showlegend=False,
                                 line=dict(color=colours[i % len(colours)], width=1.4)), row=2, col=1)
    fig.add_trace(go.Scatter(x=primary.daily.index, y=primary.daily["vix"], name="VIX",
                             line=dict(color="#886a39", width=1.2)), row=3, col=1)
    fig.add_hline(y=primary.config["panic_vix"], line_dash="dot", line_color="#c8555c", row=3, col=1)
    for i, asset in enumerate((*ASSETS, "CASH")):
        fig.add_trace(go.Scatter(x=primary.weights.index, y=primary.weights[asset], name=asset,
                                 stackgroup="weights", line=dict(width=0),
                                 fillcolor=colours[i]), row=4, col=1)
    fig.update_layout(height=1120, template="plotly_white", margin=dict(l=55, r=25, t=55, b=35),
                       font=dict(family="Arial", size=12), hovermode="x unified",
                       legend=dict(orientation="h", y=1.09), paper_bgcolor="#ffffff")
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], row=4, col=1)
    chart = fig.to_html(full_html=False, include_plotlyjs=True)

    # A compact static artifact is useful for sharing and visual verification.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    static, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1, 1]}, layout="constrained")
    for i, (name, result) in enumerate(results.items()):
        if name not in (primary.name, "spy_buy_hold", "spy_bil_no_vix", "static_vol_target"):
            continue
        axes[0].plot(result.daily.index, (1 + result.daily["return"]).cumprod(), label=name,
                     color=colours[i % len(colours)], linewidth=1.4)
        axes[1].plot(result.daily.index, drawdown(result.daily["return"]) * 100,
                     color=colours[i % len(colours)], linewidth=1)
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].set_ylabel("Drawdown (%)")
    axes[2].stackplot(primary.weights.index, primary.weights.T.to_numpy(),
                       labels=primary.weights.columns, colors=colours[:5], alpha=.85)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Allocation")
    axes[2].legend(loc="upper left", ncol=5, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    synthetic = primary.metadata.get("synthetic", False)
    title = "SYNTHETIC DATA — plumbing demonstration only" if synthetic else "VIX strategy · historical research simulation"
    static.suptitle(title, fontsize=17, fontweight="bold")
    static.savefig(folder / "overview.png", dpi=160)
    static.savefig(folder / "overview.svg")
    plt.close(static)

    metrics = performance(primary.daily)
    cards = "".join(f'<div class="card"><span>{label}</span><strong>{value}</strong></div>'
                    for label, value in (("CAGR", f'{metrics["cagr"]:.1%}'),
                    ("Sharpe · excess returns", f'{metrics["sharpe"]:.2f}'),
                    ("Annual volatility", f'{metrics["annual_vol"]:.1%}'),
                    ("Maximum drawdown", f'{metrics["max_drawdown"]:.1%}')))
    monthly = monthly_returns(primary.daily).map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    monthly.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    status = "SYNTHETIC DEMO · NO INVESTMENT EVIDENCE" if synthetic else "HISTORICAL MARKET DATA · RETROSPECTIVE RESEARCH"
    execution = primary.execution
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>VIX strategy research</title>
<style>body{{margin:0;background:#f3f5f7;color:#183045;font:15px/1.6 Arial,sans-serif}}main{{max-width:1400px;margin:auto;padding:32px}}
h1{{font-size:36px;margin:8px 0}}h2{{font-size:22px;margin-top:30px}}.status{{font-size:12px;letter-spacing:1px;color:#9b4d1c;font-weight:bold}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0}}.card,.panel{{background:white;border-radius:12px;padding:20px}}
.card span{{display:block;color:#667789;font-size:13px}}.card strong{{font-size:30px}}.panel{{margin-bottom:20px;overflow:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px;border-bottom:1px solid #e5e9ef;text-align:right;white-space:nowrap}}th:first-child{{text-align:left}}
.muted{{color:#64748b}}code{{background:#eef2f5;padding:2px 5px}}@media(max-width:750px){{main{{padding:16px}}.cards{{grid-template-columns:repeat(2,1fr)}}}}
</style><main><div class="status">{status}</div><h1>Fear regimes, measured.</h1>
<p class="muted">{escape(primary.name)} · {primary.daily.index[0].date()} — {primary.daily.index[-1].date()} · {escape(str(primary.metadata.get("source", "")))}</p>
<div class="cards">{cards}</div><div class="panel">{chart}</div>
<div class="panel"><h2>Same-period comparisons</h2><p>Click chart legends to show more variants. All models share dates and cost assumptions; their realised risk can differ.</p>
{format_comparison(table).to_html(border=0, escape=True)}
<p class="muted">corrected_original_rules is an approximation of the earlier allocation rules with repaired scheduling and thresholds, not an exact replay of the buggy original.</p></div>
<div class="panel"><h2>Monthly returns · {escape(primary.name)}</h2>{monthly.to_html(border=0)}</div>
<div class="panel"><h2>Regime attribution</h2>{regime_attribution(primary.daily).to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}")}
<p class="muted">Log-return contributions add to the total log return. Regime labels reflect the signal known before the session; attribution is not proof that a regime caused a return.</p></div>
<div class="panel"><h2>Uncertainty</h2>{ci.to_html(border=0, float_format=lambda x: f"{x:.3f}")}
<p class="muted">2.5%, median and 97.5% quantiles from a seeded 20-session block bootstrap ({bootstrap_samples} samples). Conditional on this history; not a forecast or correction for model selection.</p></div>
<div class="panel"><h2>Assumptions and audit trail</h2><p>Signals use the preceding completed session. Research fills use the next adjusted open; the QuantConnect adapter trades five minutes after the open. These are different execution models.</p>
<p>Commission: {execution["commission_bps"]:g} bps per side. Slippage: {execution["slippage_bps"]:g} bps per side. Fractional adjusted-price units; dividends are represented through price adjustment and are not added again. USD cash earns the lagged cash-rate proxy. No taxes or currency conversion.</p>
<p>Sharpe and Sortino use time-varying excess returns. Annualisation uses 252 sessions. Gross turnover counts both purchases and sales. Current drawdowns are included in underwater duration.</p>
<p>CSV exports contain daily equity, actual weights, targets, fills, costs, yearly metrics and rolling statistics. Each model has a manifest with configuration, input hashes (when loaded from disk), package versions and source hashes.</p>
<p><strong>Higher performance in this report does not establish an improvement.</strong> Use the separate walk-forward and robustness commands, validate fills in LEAN, and retain untouched data for final evaluation.</p></div></main></html>'''
    (folder / "report.html").write_text(content, encoding="utf-8")
    return folder / "report.html"
