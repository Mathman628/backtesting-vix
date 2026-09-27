"""Standalone plots, report and machine-readable comparison findings."""
from html import escape
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'outputs/research_extension/.matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .extension_study import OUT, PERIODS
from .literature_study import restore

LABELS={'treasury_60_40':'Passive SPY/IEF 60/40','passive_50_25_25':'Passive SPY/IEF/GLD 50/25/25',
 'static_risk_control':'Static + risk control','ensemble':'1/3/12-month ensemble','previous_revised':'Previous revised VIX strategy',
 'passive_five':'Passive five assets','trend_survivors':'Five-asset trend: eligible assets equal-weighted',
 'trend_slots':'Five-asset trend: fixed 20% slots','hard_vix':'Hard VIX + 12-month trend','smooth_vix':'Smooth VIX + 12-month trend',
 'unconditional':'Unconditional optimiser','hmm':'HMM optimiser','tracked_hmm':'Tracked/smoothed HMM optimiser',
 'tracked_no_penalty':'Tracked HMM, no turnover penalty'}
FAMILIES={'trend':['passive_five','trend_survivors','trend_slots'],
 'vix':['hard_vix','smooth_vix','static_risk_control','ensemble'],
 'hmm':['unconditional','hmm','tracked_hmm','tracked_no_penalty','static_risk_control']}


def build():
    hypotheses=pd.read_csv(OUT/'hypotheses.csv');fits=pd.read_csv(OUT/'hmm_fits.csv')
    common=pd.read_csv(OUT/'common_horizon/comparison.csv',index_col=0)
    original=pd.read_csv(OUT/'original_horizon/comparison.csv',index_col=0)
    takeaways=[
      f"The HMM did not improve on its simpler optimiser control in 2011–2023: excess-return Sharpe {common.loc['hmm','sharpe']:.3f} versus {common.loc['unconditional','sharpe']:.3f}. Tracking and smoothing raised it to {common.loc['tracked_hmm','sharpe']:.3f}, still below the unconditional control. Static + risk control scored {common.loc['static_risk_control','sharpe']:.3f}; passive 60/40 scored {common.loc['treasury_60_40','sharpe']:.3f} on those same dates.",
      f"Smoother VIX exposure looks primarily like a turnover improvement. Over 2007–2023 its Sharpe was {original.loc['smooth_vix','sharpe']:.3f} versus {original.loc['hard_vix','sharpe']:.3f}; annual turnover fell from {original.loc['hard_vix','gross_turnover_per_year']:.2f}x to {original.loc['smooth_vix','gross_turnover_per_year']:.2f}x. Maximum drawdown was slightly worse, not better. The hard control here uses the same 12-month trend and static base weights; it is not identical to the separate previous revised VIX strategy.",
      f"Five-asset trend reduced the matching passive portfolio's 2007–2023 drawdown. Survivor reweighting reached Sharpe {original.loc['trend_survivors','sharpe']:.3f}; fixed slots reached {original.loc['trend_slots','sharpe']:.3f}; passive five assets scored {original.loc['passive_five','sharpe']:.3f}. Neither beat static + risk control ({original.loc['static_risk_control','sharpe']:.3f}). The 2011–2023 trend Sharpes were also below the matching passive five-asset control, so the improvement was not consistent across windows."]
    takeaways.append('Every paired 95% Sharpe-difference interval in this experiment includes zero. Smoothing reduces turnover, but this run does not establish a reliable Sharpe advantage for any new treatment. The later HMM CAGR is higher than its unconditional control, but its Sharpe is slightly lower and its drawdown worse; return alone would give a misleading impression of improvement.')
    lines=['# Research extension: results and interpretation','',
      'These are new implementations evaluated on previously inspected history. They are not an untouched test set. No winning policy has been promoted into the trading strategy.',
      '', '## Main findings', '', *[t+'\n' for t in takeaways],
      '', '## What changed', '',
      'Implemented monthly five-asset trend following, a controlled smooth-versus-hard VIX overlay, and a fixed-three-state HMM adaptation with causal probability filtering, Wasserstein template matching, smoothed return estimates and turnover-aware allocation. See [the source and hypothesis register](../../EXTENSION_REGISTER.md) for implemented and deferred concepts.',
      '', 'The ordinary 60/40 and passive 50/25/25 benchmarks remain in every comparison. Five-asset trend has its own same-universe passive control. HMM policies have a same-universe, same-constraints unconditional optimiser control. Comparing HMM directly with passive 50/25/25 alone would confound allocation, risk control and regime estimation.',
      '', '## Periods and interpretation', '',
      'The original 2007–2023 period retains the financial crisis. All-policy comparisons start in January 2011 because the HMM requires 756 complete prior observations including BIL. This means the HMM comparison does not establish performance during the 2008 crisis. The already-viewed 2024–2026 period is separate. Every period starts a new account; annual tables within each period retain continuous holdings.',
      '', 'HMM monthly refits use only earlier observations, and probabilities are updated with a forward filter. Parameters are fixed for this experiment. This provides causal rolling-estimation evidence, not independence from our earlier research choices.',
      '', f'Fit diagnostics: {len(fits)} monthly fits; {int(fits.hit_iteration_limit.sum())} reached the 100-iteration limit. An iteration-limit hit is not a demonstrated converged estimate. All completed allocation optimisations passed solver-success checks; a solver failure would stop the run.',
      '', '## Results by matched period','']
    sections=[]
    for period,(start,end) in PERIODS.items():
        results=restore(OUT/period);table=pd.read_csv(OUT/period/'comparison.csv',index_col=0)
        lines += [f'### {period}: {start} to {end}','',
          '| Policy | CAGR | Excess Sharpe | Maximum drawdown | Annual gross turnover |',
          '|---|---:|---:|---:|---:|']
        for name,row in table.iterrows():
            lines.append(f'| {LABELS[name]} | {row.cagr:.2%} | {row.sharpe:.3f} | {row.max_drawdown:.2%} | {row.gross_turnover_per_year:.2f}x |')
        lines += ['']
        subset=hypotheses[hypotheses.period==period]
        comments=[]
        for _,h in subset.iterrows():
            higher=h.sharpe_difference>0
            practical=h.turnover_change<0 and h.sharpe_difference>=-.02
            criterion=(h.drawdown_improvement>0 if h.treatment in ('trend_survivors','trend_slots') else higher if h.treatment=='hmm' else practical)
            criterion_name=('smaller drawdown expectation' if h.treatment in ('trend_survivors','trend_slots') else 'higher Sharpe criterion' if h.treatment=='hmm' else 'lower turnover without losing more than .02 Sharpe criterion')
            evidence='positive' if h.pointwise_95_low>0 else 'negative' if h.pointwise_95_high<0 else 'includes zero'
            sentence=(f'{LABELS[h.treatment]} versus {LABELS[h.control]}: Sharpe difference {h.sharpe_difference:+.3f}, '
              f'paired pointwise 95% interval [{h.pointwise_95_low:+.3f}, {h.pointwise_95_high:+.3f}] ({evidence}); '
              f'drawdown improvement {h.drawdown_improvement:+.2%}; annual turnover change {h.turnover_change:+.2f}x. '
              f'The {criterion_name} is {"met" if criterion else "not met"}; this is descriptive support, not proof.')
            comments.append(sentence);lines += ['- '+sentence]
        lines += ['']
        charts=[]
        for family,names in FAMILIES.items():
            names=[n for n in names if n in results]
            if len(names)<2:continue
            fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
            for name in names:
                d=results[name].daily;wealth=(1+d['return']).cumprod()
                peak=wealth.cummax().clip(lower=1)
                axes[0].plot(wealth.index,wealth,label=LABELS[name],lw=1.3)
                axes[1].plot(wealth.index,100*(wealth/peak-1),lw=1.1)
            axes[0].set_yscale('log');axes[0].set_ylabel('Growth of $1 (log scale)');axes[0].legend(fontsize=8)
            axes[1].set_ylabel('Drawdown (%)');axes[0].set_title(f'{family.upper()}: {start} to {end}, after costs')
            for ax in axes:ax.grid(alpha=.2)
            fig.tight_layout();filename=f'{period}_{family}.png';fig.savefig(OUT/filename,dpi=150);plt.close(fig)
            charts.append(filename)
        annual=pd.DataFrame({LABELS[name]:(1+result.daily['return']).groupby(result.daily.index.year).prod()-1 for name,result in results.items()}).T
        fig,ax=plt.subplots(figsize=(max(9,len(annual.columns)*.7),max(5,len(annual)*.42)))
        im=ax.imshow(100*annual.to_numpy(),cmap='RdYlGn',vmin=-30,vmax=30,aspect='auto')
        ax.set_xticks(range(len(annual.columns)),annual.columns,rotation=45,ha='right');ax.set_yticks(range(len(annual.index)),annual.index,fontsize=8)
        for i in range(len(annual)):
            for j in range(len(annual.columns)):ax.text(j,i,f'{100*annual.iloc[i,j]:.0f}',ha='center',va='center',fontsize=7)
        ax.set_title('Calendar-year net returns (%); boundary years can be partial')
        fig.colorbar(im,ax=ax,label='Return (%)');fig.tight_layout()
        filename=f'{period}_annual.png';fig.savefig(OUT/filename,dpi=150);plt.close(fig);charts.append(filename)
        annual.to_csv(OUT/period/'annual_returns.csv')
        display=table[['cagr','sharpe','annual_vol','max_drawdown','gross_turnover_per_year']].copy()
        display.index=[LABELS[n] for n in display.index]
        for col in ['cagr','annual_vol','max_drawdown']:display[col]=display[col].map(lambda x:f'{x:.2%}')
        display['sharpe']=display.sharpe.map(lambda x:f'{x:.3f}')
        display['gross_turnover_per_year']=display.gross_turnover_per_year.map(lambda x:f'{x:.2f}x')
        sections.append(f'<section id="{period}"><h2>{escape(period)}: {start} to {end}</h2>'+display.to_html()+
          ''.join(f'<p>{escape(c)}</p>' for c in comments)+''.join(f'<img src="{f}" alt="{f}">' for f in charts)+'</section>')
    # Regime labels are stable template identifiers, not ex-post bull/bear labels.
    probs=pd.read_csv(OUT/'regime_probabilities.csv',parse_dates=['date']).set_index('date')
    monthly=probs.filter(like='probability').resample('ME').mean()
    fig,ax=plt.subplots(figsize=(12,3));ax.stackplot(monthly.index,monthly.to_numpy().T,labels=['Template 0','Template 1','Template 2'])
    ax.set_ylim(0,1);ax.set_title('Mean monthly predicted HMM probabilities; labels have no fixed bull/bear meaning');ax.legend(loc='upper left',ncol=3)
    fig.tight_layout();fig.savefig(OUT/'regime_probabilities.png',dpi=150);plt.close(fig)
    attribution=pd.read_csv(OUT/'common_horizon/attribution.csv',index_col='policy')
    names=['static_risk_control','unconditional','hmm','tracked_hmm']
    averages=pd.DataFrame({LABELS[n]:pd.read_csv(OUT/'common_horizon'/n/'weights.csv').drop(columns='date').mean() for n in names}).T
    averages[['SPY','IEF','GLD','BIL','CASH']].mul(100).plot.barh(stacked=True,figsize=(10,4))
    plt.xlabel('Average closing portfolio weight (%)');plt.ylabel('');plt.title('2011–2023: exposure differences behind the returns');plt.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=5)
    plt.tight_layout();plt.savefig(OUT/'common_average_weights.png',dpi=150);plt.close()
    exposure_text=(f"The common-period accounting attribution helps explain the HMM gap without proving its cause. Its gold sleeve contributed {100*attribution.loc['hmm','contribution_GLD']:.1f} percentage points of initial capital, versus {100*attribution.loc['unconditional','contribution_GLD']:.1f} for the unconditional optimiser. Cumulative cost contributions were {100*attribution.loc['hmm','contribution_cost']:.1f} and {100*attribution.loc['unconditional','contribution_cost']:.1f} points respectively. These are wealth-linked contributions over the entire period, not annual returns or estimates of causal alpha. Average allocations in common_average_weights.png show that the models also held different exposures.")
    lines += ['## Exposure and trading costs','',exposure_text,'','## Reading the graphs','',
      'Wealth graphs use a log scale and identical initial capital within each period. The drawdown panels measure loss from each strategy’s running high, including initial capital. Compare curves within a family first: the five-asset chart isolates trend timing; the VIX chart isolates smoothing; the HMM chart separates state estimation and moment stabilisation from the unconditional optimiser. Static risk control is a useful simpler reference, but does not share the optimiser’s objective.',
      '', 'The probability chart averages daily next-state probabilities by month. Its colours identify persistent templates, not economically certified bull/bear regimes. High confidence does not establish forecasting accuracy. See weights.csv, yearly_metrics.csv and attribution.csv in each policy/period directory to check concentration, crisis behaviour and favourable asset exposure.',
      '', '## What the experiment can and cannot establish','',
      'A lower drawdown can result from holding more cash or bonds, rather than skill. A higher Sharpe in the short later period can reflect favourable equity/gold trends. The export includes exact wealth-linked asset contributions, but this is accounting attribution, not a causal estimate of timing alpha. Dollar cost differences also depend on portfolio growth; use turnover alongside costs.',
      '', 'The HMM uses noisy conditional means, a limited universe and predetermined hyperparameters. Fixed three states, diagonal feature covariance and smoothed templates depart from the source. Features describe yesterday’s market; a state is not a guarantee of tomorrow’s returns. Template matching alone cannot improve returns because consistent state permutations leave a mixture unchanged; our tracked treatment additionally smooths return moments.',
      '', 'Paired 20-day block-bootstrap intervals use 2,000 resamples. They are pointwise and assume resampled historical dependence is informative; they do not adjust for all prior trials, parameter choices or changing market structure. A positive point estimate with an interval crossing zero remains uncertain. Robust confirmation would require precommitting a policy and observing genuinely unseen future data, plus checking alternative transaction costs and estimation windows in a separately registered study.',
      '', 'No settings were tuned to rescue an unfavourable result. Adaptive state-count selection, broader assets and the unread sector-rotation strategy remain deferred. The original frozen study and live strategy were not changed.']
    (OUT/'ANALYSIS.md').write_text('\n'.join(lines),encoding='utf-8')
    intro='New research adaptation; all dates previously inspected. Read ANALYSIS.md and the source register before interpreting performance. Select a matched period below. The original horizon includes 2008; HMM comparisons begin in 2011 after the required BIL estimation history.'
    html='<!doctype html><html><head><meta charset="utf-8"><title>Trend and regime research extension</title><style>body{font:16px system-ui;max-width:1280px;margin:35px auto;padding:0 24px;color:#172b3a}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}img{width:100%;height:auto}select{padding:10px}p{line-height:1.55}section{margin-top:30px}</style></head><body><h1>Trend and regime research extension</h1><p>'+escape(intro)+'</p><p><a href="ANALYSIS.md">Full analysis</a> · <a href="../../EXTENSION_REGISTER.md">Sources, hypotheses and omissions</a> · <a href="hypotheses.csv">Paired comparisons</a></p><select onchange="document.querySelectorAll(\'section\').forEach(s=>s.hidden=s.id!==this.value)">'+''.join(f'<option value="{p}">{p}: {a} to {b}</option>' for p,(a,b) in PERIODS.items())+'</select>'+''.join(sections)+'<h2>Causal regime probabilities</h2><img src="regime_probabilities.png" alt="Monthly mean state probabilities"><script>document.querySelectorAll("section").forEach((s,i)=>s.hidden=i!==0)</script></body></html>'
    (OUT/'report.html').write_text(html,encoding='utf-8')
    html=html.replace('<select onchange=', '<h2>Main findings</h2>'+''.join('<p>'+escape(t)+'</p>' for t in takeaways)+'<select onchange=',1)
    html=html.replace('<h2>Causal regime probabilities</h2>', '<h2>Exposure and cost attribution</h2><p>'+escape(exposure_text)+'</p><img src="common_average_weights.png" alt="Average portfolio allocations"><h2>Causal regime probabilities</h2>')
    (OUT/'report.html').write_text(html,encoding='utf-8')
    print('Report:',OUT/'report.html',flush=True)


if __name__=='__main__':build()
