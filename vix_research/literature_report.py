"""Readable findings and figures for the frozen study; never changes a strategy."""
from pathlib import Path
from html import escape
import json
import os
import numpy as np
import pandas as pd
from .literature_study import restore, PERIODS
from .literature_analysis import BENCHMARK, HYPOTHESES

LABELS = {"factor_t0_r0_v0":"Static allocation", "factor_t0_r0_v1":"Static + VIX",
 "factor_t0_r1_v0":"Static + risk control", "factor_t0_r1_v1":"Static + risk + VIX",
 "factor_t1_r0_v0":"Trend only", "factor_t1_r0_v1":"Trend + VIX",
 "factor_t1_r1_v0":"Trend + risk", "factor_t1_r1_v1":"Trend + risk + VIX",
 "excess12":"12-month excess trend", "ensemble":"1/3/12-month ensemble",
 "ensemble_band":"Ensemble + wider bands", "ensemble_partial":"Ensemble + partial trades",
 "ensemble_reentry":"Ensemble + re-entry cap", "blend_80_20":"80/20 static/trend blend",
 "inverse_vol_ensemble":"Inverse-volatility ensemble", "regime_ensemble":"Original regime map + ensemble",
 "original_revised":"Previous revised strategy", "original_sma":"Previous SMA-only variant",
 "original_static_vol":"Previous static benchmark"}
SHORT_H = {"H1":"Trend signal", "H2":"Risk cap", "H3":"VIX overlay", "H4":"Cash hurdle",
 "H5":"Multiple horizons", "H6":"Wider bands", "H7":"Partial trades", "H8":"Re-entry cap",
 "H9":"80/20 blend", "H10":"Inverse-vol weights", "H11":"Original regime map"}


def build_report(folder):
    folder=Path(folder)
    if not (folder/"test_complete.json").exists():
        raise ValueError("Complete the frozen test before producing findings")
    selection=json.loads((folder/"selection.json").read_text())
    selected=selection["selected"]
    results={split:restore(folder/split) for split in PERIODS}
    tables={s:pd.read_csv(folder/s/"comparison.csv",index_col=0) for s in PERIODS}
    hypotheses=pd.concat([pd.read_csv(folder/s/"hypotheses.csv") for s in PERIODS],ignore_index=True)
    hypotheses.to_csv(folder/"hypothesis_outcomes.csv",index=False)
    out=folder/"figures";out.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR",str((folder/".matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.titlesize":10,
                         "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    focus=list(dict.fromkeys([BENCHMARK,"original_revised","ensemble",selected]))
    colors={n:c for n,c in zip(focus,["#4d667c","#b45d35","#087f8c","#8870b2"])}
    def save(fig,name):
        for ext in ("pdf","png"):
            fig.savefig(out/f"{name}.{ext}",dpi=180,bbox_inches="tight",facecolor="white")
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(8,3.1),sharey=True,layout="constrained")
    for ax,split in zip(axes,PERIODS):
        vals=tables[split].loc[focus,"sharpe"]
        ax.bar(np.arange(len(focus)),vals,color=[colors[n] for n in focus])
        ax.set_xticks(range(len(focus)),[str(i+1) for i in range(len(focus))])
        ax.axhline(0,color="#444",lw=.5);ax.set_title(split.title());ax.grid(axis="y",alpha=.15)
        for i,v in enumerate(vals):ax.annotate(f"{v:.2f}",(i,v),xytext=(0,4 if v>=0 else -12),textcoords="offset points",ha="center",fontsize=8)
    axes[0].set_ylabel("Excess-return Sharpe")
    fig.supxlabel("   |   ".join(f"{i+1}: {LABELS[n]}" for i,n in enumerate(focus)),fontsize=7)
    save(fig,"01_chronological")
    fig,axes=plt.subplots(2,1,figsize=(8,4.8),sharex=True,layout="constrained")
    for n in focus:
        r=results["test"][n].daily["return"];growth=(1+r).cumprod()
        axes[0].plot(r.index,growth*100,label=LABELS[n],color=colors[n],lw=1.2)
        axes[1].plot(r.index,(growth/growth.cummax().clip(lower=1)-1)*100,color=colors[n],lw=1)
    axes[0].legend(fontsize=7,loc="upper left",frameon=False,ncol=2)
    axes[0].set_ylabel("Wealth (initial = 100)");axes[1].set_ylabel("Drawdown (%)")
    axes[0].set_title("Frozen historical test: 2024-01-02 to 2026-09-23",loc="left")
    for ax in axes:ax.grid(alpha=.15)
    save(fig,"02_test_performance")
    h=hypotheses[hypotheses.split=="test"].set_index("hypothesis")
    fig,ax=plt.subplots(figsize=(8,4.2),layout="constrained")
    for i,(key,row) in enumerate(h.iterrows()):
        ax.plot([row.sharpe_ci_low,row.sharpe_ci_high],[i,i],color="#8299aa",lw=2)
        ax.scatter(row.sharpe_difference,i,color="#087f8c",s=25,zorder=3)
    ax.set_yticks(range(len(h)),[f"{key}: {SHORT_H[key]}" for key in h.index]);ax.invert_yaxis()
    ax.axvline(0,color="#b45d35",ls="--",lw=1);ax.grid(axis="x",alpha=.15)
    ax.set_xlabel("Treatment minus control Sharpe; paired 95% pointwise interval")
    ax.set_title("Paired uncertainty: 2,000 circular 20-session block resamples",loc="left")
    save(fig,"03_paired_hypotheses")
    stress=pd.read_csv(folder/"test/cost_stress.csv")
    episodes=pd.concat([pd.read_csv(folder/s/"episodes.csv") for s in PERIODS],ignore_index=True)
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),layout="constrained")
    cost_focus=list(dict.fromkeys([BENCHMARK,"ensemble","ensemble_band",selected]))
    for n in cost_focus:
        d=stress[stress.model==n]
        if not d.empty:axes[0].plot(d.slippage_bps,d.sharpe,marker="o",ms=3,label=LABELS[n],color=colors.get(n,"#8870b2"))
    axes[0].set_xlabel("Slippage per side (bps)");axes[0].set_ylabel("Test Sharpe")
    axes[0].set_title("(a) Fixed-policy cost sensitivity",loc="left")
    axes[0].legend(fontsize=6.5,frameon=False);axes[0].set_xticks([0,5,10,25])
    recoveries=["GFC_recovery","COVID_recovery","Recovery_2023"]
    for j,n in enumerate(focus):
        d=episodes[episodes.model==n].set_index("episode").reindex(recoveries)
        axes[1].bar(np.arange(3)+(j-(len(focus)-1)/2)*.19,d.return_gap*100,width=.18,color=colors[n],label=LABELS[n])
    axes[1].set_xticks(range(3),["2009","2020","2023"]);axes[1].set_ylabel("Return gap vs static (pp)")
    axes[1].set_title("(b) Ex-post recovery windows",loc="left");axes[1].axhline(0,color="#444",lw=.5)
    axes[1].legend(fontsize=6.2,frameon=False,loc="lower right")
    for ax in axes:ax.grid(axis="y",alpha=.15)
    save(fig,"04_costs_recovery")
    fig,axes=plt.subplots(1,2,figsize=(8,3.5),layout="constrained")
    att=pd.read_csv(folder/"test"/selected/"attribution.csv",index_col=0)
    axes[0].bar(att.index.str.replace("contribution_",""),att.linked_total_return_contribution*100,color="#087f8c")
    axes[0].set_title("(a) Selected model: linked contributions",loc="left",fontsize=9)
    axes[0].set_ylabel("Contribution to total return (pp)");axes[0].axhline(0,color="#444",lw=.5)
    capture=pd.concat([pd.read_csv(folder/s/"market_capture.csv") for s in PERIODS])
    for a,c in zip(["SPY","IEF","GLD"],["#087f8c","#597ba4","#d2aa47"]):
        d=capture[(capture.model=="ensemble")&(capture.asset==a)]
        axes[1].scatter(d.absolute_market_sharpe,d.contribution_sharpe,s=15,alpha=.7,color=c,label=a)
    axes[1].set_title("(b) Ensemble capture: descriptive only",loc="left",fontsize=9)
    axes[1].set_xlabel("Absolute annual market Sharpe");axes[1].set_ylabel("Asset contribution mean / volatility")
    axes[1].legend(fontsize=7,frameon=False)
    save(fig,"05_attribution_capture")

    # Write a complete results record; the human-edited report can cite these values.
    md=["# Frozen literature study: results", "",
        f"Selection frozen before test execution: **{LABELS[selected]}** (`{selected}`).",
        f"Development shortlist: {', '.join(selection['shortlist'])}. Validation fallback: {selection['validation_fallback']}.",
        "", "Historical test history was already viewed in the earlier project. This is chronological retrospective evidence, not a new untouched holdout.",
        "", "## Selected model and benchmark"]
    for split,t in tables.items():
        a,b=t.loc[selected],t.loc[BENCHMARK]
        md.append(f"- {split}: selected CAGR {a.cagr:.2%}, volatility {a.annual_vol:.2%}, Sharpe {a.sharpe:.3f}, maximum drawdown {a.max_drawdown:.2%}; benchmark Sharpe {b.sharpe:.3f}, drawdown {b.max_drawdown:.2%}.")
    md += ["", "## Expected direction versus observed results", "",
           "Criteria are frozen in SOURCE_REGISTER.md. Support below is descriptive; confidence intervals concern Sharpe differences only, not the drawdown or cost criteria.", "",
           "| Hypothesis | Development | Validation | Test | Test Sharpe difference [95% interval] |",
           "|---|---|---|---|---|"]
    for key,_,_,_ in HYPOTHESES:
        d=hypotheses[hypotheses.hypothesis==key].set_index("split");r=d.loc["test"]
        status=["Direction supported" if bool(d.loc[s,"descriptive_support"]) else "Direction not supported" for s in PERIODS]
        md.append(f"| {key}: {SHORT_H[key]} | {' | '.join(status)} | {r.sharpe_difference:+.3f} [{r.sharpe_ci_low:+.3f}, {r.sharpe_ci_high:+.3f}] |")
    md += ["", "## Interpretation rules", "",
       "- Do not promote the best test-period performer: selection was fixed using earlier periods.",
       "- A pointwise interval crossing zero is inconclusive, not proof of equivalence. Eleven comparisons and earlier exploration require restraint.",
       "- DSR probabilities are sensitivity diagnostics using monthly IID approximations and 19/50/100 assumed trials. Unknown earlier trials and correlated models prevent a certified global correction.",
       "- Asset contributions reconcile to portfolio returns; linked contributions sum to total growth. Contribution correlations differ from correlations of standalone fully invested strategies.",
       "- Cash-drag figures are signed, ex-post daily-sum counterfactuals using prior-close weights. They are not compounded foregone returns, executable alpha or isolated policy effects.",
       "- The market-capture graph is inspired by Babu et al.; it is not their exact fitted decomposition. Asset contribution Sharpe does not deduct a standalone cash hurdle.",
       "- Crisis windows are retrospective descriptions, not information used by the trading policy.",
       "- The original QuantConnect default is unchanged. New variants are research implementations only; actual LEAN fills remain unverified.",
       "", "## Files", "",
       "Open report.html for the interactive view. SOURCE_REGISTER.md records every source and omission; protocol.json fixes expectations and hashes; selection.json records the pre-test decision; hypothesis_outcomes.csv contains all split comparisons. Each split/model folder has daily returns, signals, weights, trades, attribution and cost triggers."]
    (folder/"FINDINGS.md").write_text("\n".join(md)+"\n",encoding="utf-8")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.06,
        subplot_titles=("Historical test wealth after costs", "Drawdown", "Frozen selected model: actual allocations"))
    for n,r in results["test"].items():
        g=(1+r.daily["return"]).cumprod();visible=True if n in focus else "legendonly"
        fig.add_trace(go.Scatter(x=g.index,y=g*100,name=LABELS[n],legendgroup=n,visible=visible),row=1,col=1)
        fig.add_trace(go.Scatter(x=g.index,y=g/g.cummax().clip(lower=1)-1,name=LABELS[n],legendgroup=n,showlegend=False,visible=visible),row=2,col=1)
    for asset in results["test"][selected].weights:
        w=results["test"][selected].weights[asset]
        fig.add_trace(go.Scatter(x=w.index,y=w,name=asset,stackgroup="weights",line=dict(width=0)),row=3,col=1)
    fig.update_yaxes(tickformat=".0%",row=2,col=1);fig.update_yaxes(tickformat=".0%",row=3,col=1)
    fig.update_layout(height=1000,template="plotly_white",hovermode="x unified",legend=dict(orientation="h",y=-.08))
    charts=fig.to_html(full_html=False,include_plotlyjs=True)
    h_display=hypotheses.copy()
    h_display["comparison"]=[f"{LABELS[a]} vs {LABELS[b]}" for a,b in zip(h_display.treatment,h_display.control)]
    html=["<!doctype html><html><meta charset='utf-8'><title>Literature study: frozen hypotheses and results</title>",
       "<style>body{font:15px/1.6 Arial;color:#183045;background:#f3f5f7;margin:0}main{max-width:1300px;margin:auto;padding:30px}section{background:white;padding:24px;margin:20px 0;overflow:auto;border-radius:10px}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border-bottom:1px solid #ddd;white-space:nowrap;text-align:right}img{max-width:100%}a{color:#087f8c}</style><main>",
       "<h1>Literature study: expectations tested</h1>",
       f"<p><strong>Frozen selection: {escape(LABELS[selected])}.</strong> Historical test: January 2024–September 2026. This period was previously viewed; no claim of an untouched holdout.</p>",
       "<p><a href='ANALYSIS.md'>Interpretation and recommendations</a> · <a href='FINDINGS.md'>Detailed findings</a> · <a href='SOURCE_REGISTER.md'>Sources and implementation ledger</a> · <a href='protocol.json'>Frozen protocol</a> · <a href='selection.json'>Pre-test selection</a> · <a href='hypothesis_outcomes.csv'>All hypothesis results</a></p>",
       "<section>"+charts+"</section>"]
    for split,t in tables.items():
        display=t[["cagr","annual_vol","sharpe","max_drawdown","gross_turnover_per_year"]].rename(index=LABELS)
        html.append(f"<section><h2>{split.title()} results</h2><p>{PERIODS[split][0]} to {PERIODS[split][1]}; independently starts in cash.</p>"+display.to_html(float_format=lambda x:f"{x:.4f}")+"</section>")
    html.append("<section><h2>Frozen hypotheses</h2><p>Descriptive support uses the predeclared criterion. Sharpe intervals are pointwise, not multiplicity-adjusted.</p>"+h_display[["hypothesis","split","comparison","criterion","descriptive_support","sharpe_difference","sharpe_ci_low","sharpe_ci_high"]].to_html(index=False,float_format=lambda x:f"{x:.3f}")+"</section>")
    captions={"01_chronological":"Sharpe across independent periods; the model label selected by development/validation stays fixed.",
              "02_test_performance":"Separate historical test, not a continuation of development capital.",
              "03_paired_hypotheses":"Sharpe differences resample identical dates across policies, preserving dependence in 20-session blocks.",
              "04_costs_recovery":"Cost scenarios do not change selected parameters. Recovery windows are defined retrospectively.",
              "05_attribution_capture":"Linked contributions explain terminal growth; capture scatter is descriptive and not an exact paper replication."}
    for name,caption in captions.items():
        html.append(f"<section><img src='figures/{name}.png'><p>{caption}</p></section>")
    html.append("<section><h2>Evidence limits</h2><p>Previous test exposure, selection bias, short test history, estimated costs and unverified live execution remain. See FINDINGS.md for accounting, cash-drag and approximate DSR definitions. No variant has been deployed to a broker.</p></section></main></html>")
    (folder/"report.html").write_text("\n".join(html),encoding="utf-8")
    print(f"Open {folder/'report.html'}",flush=True)
