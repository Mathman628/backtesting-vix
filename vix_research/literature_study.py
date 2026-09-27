"""Frozen chronological study. Run via `python -m vix_research literature`.

Selection is written after development/validation and before historical test.
Existing study folders cannot be silently re-frozen over completed experiments.
"""
from pathlib import Path
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

from .core import StrategyConfig
from .backtest import run_backtest, prepare_snapshots, ExecutionConfig, BacktestResult
from .data import load_data
from .analytics import comparison_table, performance, yearly_metrics
from .report import save_json
from .literature_strategy import variants, research_snapshots, ResearchEngine
from .literature_analysis import (BENCHMARK, HYPOTHESES, paired_sharpes,
    compare_hypotheses, deflated_sharpe, attribution, episode_analysis, market_capture)

ROOT = Path(__file__).resolve().parent
PERIODS = {"development": ("2007-08-29", "2018-12-31"),
           "validation": ("2019-01-01", "2023-12-31"),
           "test": ("2024-01-01", "2026-09-23")}
SOURCE_FILES = ("core.py", "backtest.py", "data.py", "analytics.py",
                "literature_strategy.py", "literature_analysis.py", "literature_study.py",
                "LITERATURE_REGISTER.md")


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def definitions():
    return {**{n: c.to_dict() for n,c in variants().items()},
            "original_revised": StrategyConfig().to_dict(),
            "original_sma": replace(StrategyConfig(), equity_filter="sma").to_dict(),
            "original_static_vol": StrategyConfig().to_dict()}


def freeze(folder, data_folder):
    folder, data_folder = Path(folder), Path(data_folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "protocol.json"
    if path.exists():
        verify(folder, data_folder)
        return json.loads(path.read_text())
    protocol = dict(created_utc=utc(), periods=PERIODS, configurations=definitions(),
        benchmark=BENCHMARK, hypotheses=HYPOTHESES, execution=ExecutionConfig().__dict__,
        selection="Top three development Sharpes, then best validation Sharpe; validation benchmark fallback; lower turnover/name tie-break",
        test_status="Previously viewed historical period, NOT untouched or prospective",
        expected_outcomes=(ROOT/"LITERATURE_REGISTER.md").read_text(encoding="utf-8"),
        source_sha256={f:digest(ROOT/f) for f in SOURCE_FILES},
        input_sha256={f:digest(data_folder/f) for f in ("prices.csv", "risk_free.csv")},
        bootstrap=dict(samples=2000,block=20,seed=20260926,interval="pointwise 95%; no familywise correction"),
        historical_trial_count="Unknown beyond this family's 19 policies; DSR 19/50/100 sensitivity")
    save_json(path, protocol)
    (folder/"SOURCE_REGISTER.md").write_text(protocol["expected_outcomes"], encoding="utf-8")
    print(f"Frozen protocol before market simulations: {path}", flush=True)
    return protocol


def verify(folder, data_folder):
    protocol=json.loads((Path(folder)/"protocol.json").read_text())
    for f, value in protocol["source_sha256"].items():
        if digest(ROOT/f) != value:
            raise ValueError(f"Frozen implementation changed: {f}. Use a new study folder and document why.")
    for f,value in protocol["input_sha256"].items():
        if digest(Path(data_folder)/f) != value:
            raise ValueError(f"Frozen input changed: {f}")
    if protocol["configurations"] != json.loads(json.dumps(definitions())):
        raise ValueError("Configuration definitions changed")
    return protocol


def persist(results, folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    comparison_table(results).to_csv(folder/"comparison.csv")
    for name,r in results.items():
        dest=folder/name;dest.mkdir(exist_ok=True)
        for key in ("daily","weights","signals","trades"):
            getattr(r,key).to_csv(dest/f"{key}.csv", index=key!="trades")
        yearly_metrics(r.daily).to_csv(dest/"yearly_metrics.csv")
        save_json(dest/"manifest.json",dict(name=name,config=r.config,execution=r.execution,metadata=r.metadata,metrics=performance(r.daily)))


def restore(folder):
    results={}
    for name in pd.read_csv(Path(folder)/"comparison.csv",index_col=0).index:
        dest=Path(folder)/name
        info=json.loads((dest/"manifest.json").read_text())
        frames={key:pd.read_csv(dest/f"{key}.csv",parse_dates=["date"]).set_index("date") for key in ("daily","weights","signals")}
        trades=pd.read_csv(dest/"trades.csv",parse_dates=["date"])
        results[name]=BacktestResult(name,frames["daily"],frames["weights"],trades,frames["signals"],info["config"],info["execution"],info["metadata"])
    return results


def evaluate(data, period, execution=None):
    start,end=PERIODS[period]
    base=prepare_snapshots(data,StrategyConfig())
    cache={};results={}
    for name,cfg in variants().items():
        if cfg.signal not in cache:
            cache[cfg.signal]=research_snapshots(data,cfg,base)
        print(f"{period}: {name}",flush=True)
        results[name]=run_backtest(data,cfg,execution,start,end,name,snapshots=cache[cfg.signal],engine=ResearchEngine(cfg))
    for name,cfg,baseline in [("original_revised",StrategyConfig(),None),
                              ("original_sma",replace(StrategyConfig(),equity_filter="sma"),None),
                              ("original_static_vol",StrategyConfig(),"static_vol")]:
        print(f"{period}: {name}",flush=True)
        results[name]=run_backtest(data,cfg,execution,start,end,name,baseline,base)
    return results


def analyse(results, data, folder, period):
    folder=Path(folder)
    table=comparison_table(results)
    draws=paired_sharpes(results)
    compare_hypotheses(table,draws,period).to_csv(folder/"hypotheses.csv",index=False)
    comparisons=[]
    for name in results:
        low,high=np.nanquantile(draws[name]-draws[BENCHMARK],[.025,.975])
        comparisons.append(dict(model=name,control=BENCHMARK,sharpe_difference=table.loc[name,"sharpe"]-table.loc[BENCHMARK,"sharpe"],ci_low=low,ci_high=high))
    pd.DataFrame(comparisons).to_csv(folder/"paired_vs_benchmark.csv",index=False)
    deflated_sharpe(results,list(results)).to_csv(folder/"dsr_sensitivity.csv",index=False)
    rows=[]
    for name,r in results.items():
        summary,corr,error=attribution(r)
        summary.to_csv(folder/name/"attribution.csv")
        corr.to_csv(folder/name/"contribution_correlations.csv")
        rows.append(dict(model=name,max_daily_reconciliation_error=error,
                         linked_sum=summary.linked_total_return_contribution.sum(),
                         total_return=table.loc[name,"total_return"]))
        r.daily.groupby("reason")[["cost","turnover","traded"]].sum().to_csv(folder/name/"cost_by_trigger.csv")
    pd.DataFrame(rows).to_csv(folder/"accounting_checks.csv",index=False)
    episode_analysis(results,data).to_csv(folder/"episodes.csv",index=False)
    market_capture(results,data).to_csv(folder/"market_capture.csv",index=False)


def rank(table):
    return table.sort_values(["sharpe","gross_turnover_per_year"],ascending=[False,True],kind="stable").index.tolist()


def develop(folder,data_folder):
    folder=Path(folder);protocol=verify(folder,data_folder)
    if (folder/"selection.json").exists():
        print("Development and selection already recorded; reusing them.",flush=True)
        return
    data=load_data(data_folder)
    tables={}
    for period in ("development","validation"):
        results=evaluate(data,period)
        persist(results,folder/period)
        analyse(results,data,folder/period,period)
        tables[period]=comparison_table(results)
    shortlist=rank(tables["development"].sort_index())[:3]
    selected=rank(tables["validation"].loc[shortlist].sort_index())[0]
    fallback=tables["validation"].loc[selected,"sharpe"] <= tables["validation"].loc[BENCHMARK,"sharpe"]
    if fallback:
        selected=BENCHMARK
    save_json(folder/"selection.json",dict(frozen_utc=utc(),shortlist=shortlist,selected=selected,
        benchmark=BENCHMARK,validation_fallback=bool(fallback),
        protocol_sha256=digest(folder/"protocol.json"),
        development_comparison_sha256=digest(folder/"development/comparison.csv"),
        validation_comparison_sha256=digest(folder/"validation/comparison.csv"),
        selected_configuration=protocol["configurations"][selected],
        statement="Selected before this study's test execution; historical test data previously viewed in earlier research."))
    print(f"Frozen selection: {selected}; shortlist: {shortlist}",flush=True)


def test(folder,data_folder):
    folder=Path(folder);verify(folder,data_folder)
    selection_path=folder/"selection.json"
    if not selection_path.exists():
        raise ValueError("Run development/validation before test")
    selection=json.loads(selection_path.read_text())
    if selection["protocol_sha256"]!=digest(folder/"protocol.json"):
        raise ValueError("Protocol does not match selection")
    for period in ("development","validation"):
        if selection[f"{period}_comparison_sha256"] != digest(folder/period/"comparison.csv"):
            raise ValueError("Selection inputs were changed")
    if (folder/"test_complete.json").exists():
        print("Historical test already completed; no retuning or rerun.",flush=True)
        return
    started=utc();selection_hash=digest(selection_path)
    data=load_data(data_folder)
    results=evaluate(data,"test")
    persist(results,folder/"test")
    analyse(results,data,folder/"test","test")
    selected=selection["selected"]
    rows=[]
    for block in (5,20,60):
        if selected==BENCHMARK:
            low=high=0.
        else:
            draws=paired_sharpes({selected:results[selected],BENCHMARK:results[BENCHMARK]},block=block)
            low,high=np.nanquantile(draws[selected]-draws[BENCHMARK],[.025,.975])
        rows.append(dict(model=selected,block=block,ci_low=low,ci_high=high))
    pd.DataFrame(rows).to_csv(folder/"test/selected_block_sensitivity.csv",index=False)
    # Diagnostic cost stress on fixed policies. Nothing is selected from this.
    names=list(dict.fromkeys([selected,BENCHMARK,"ensemble","ensemble_band","ensemble_partial","original_revised"]))
    base=prepare_snapshots(data,StrategyConfig());cache={};rows=[]
    for bps in (0,5,10,25):
        for name in names:
            cfg=variants().get(name,replace(StrategyConfig(),equity_filter="sma") if name=="original_sma" else StrategyConfig())
            if name in variants():
                if cfg.signal not in cache:
                    cache[cfg.signal]=research_snapshots(data,cfg,base)
                r=run_backtest(data,cfg,ExecutionConfig(slippage_bps=bps),*PERIODS["test"],name,snapshots=cache[cfg.signal],engine=ResearchEngine(cfg))
            else:
                r=run_backtest(data,cfg,ExecutionConfig(slippage_bps=bps),*PERIODS["test"],name,
                    baseline="static_vol" if name=="original_static_vol" else None,snapshots=base)
            rows.append(dict(model=name,slippage_bps=bps,**performance(r.daily)))
    pd.DataFrame(rows).to_csv(folder/"test/cost_stress.csv",index=False)
    save_json(folder/"test_complete.json",dict(started_utc=started,completed_utc=utc(),
        selected=selected,selection_sha256=selection_hash,protocol_sha256=digest(folder/"protocol.json"),
        statement="Frozen historical test completed; no performance-driven parameter revisions",
        input_end=str(data.closes.index[-1].date())))
    print(comparison_table(results)[["cagr","sharpe","max_drawdown","gross_turnover_per_year"]].round(4).to_string(),flush=True)


def run(folder="vix_research/outputs/literature",data_folder="vix_research/data/market",stage="all"):
    if stage in ("freeze","all"):
        freeze(folder,data_folder)
    if stage in ("develop","all"):
        develop(folder,data_folder)
    if stage in ("test","all"):
        test(folder,data_folder)
    if stage in ("report","all"):
        from .literature_report import build_report
        build_report(folder)
