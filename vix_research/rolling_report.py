"""Presentation only for continuous annual walk-forward results."""
from pathlib import Path
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from .rolling_evaluation import LABELS


def build(folder):
    p=Path(folder)
    summary=pd.read_csv(p/'summary.csv')
    annual=pd.read_csv(p/'annual.csv')
    quarterly=pd.read_csv(p/'quarterly.csv')
    schedule=pd.read_csv(p/'schedule.csv')
    paired=pd.read_csv(p/'paired_comparisons.csv')
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=('Continuous evaluation wealth, initial 100','Drawdown (%)'))
    colors=['#a64c38','#303030','#a6802c','#486f9c','#008b8b','#9672ac']
    names=list(summary.model.unique())
    for n,color in zip(names,colors):
        d=pd.read_csv(p/'continuous'/n/'daily.csv',parse_dates=['date']).set_index('date')
        wealth=(1+d['return']).cumprod()
        for row,y in [(1,wealth*100),(2,(wealth/wealth.cummax().clip(lower=1)-1)*100)]:
            fig.add_trace(go.Scatter(x=d.index,y=y,name=LABELS[n],legendgroup=n,line_color=color,showlegend=row==1),row=row,col=1)
    fig.add_vline(x=pd.Timestamp('2024-01-01').timestamp()*1000,line_dash='dash',line_color='#999')
    fig.update_layout(template='plotly_white',height=800,legend=dict(orientation='h',y=-.1),margin=dict(b=140))
    bars=go.Figure()
    for n,color in zip(names[:3],colors[:3]):
        d=annual[annual.model==n]
        bars.add_trace(go.Bar(x=d.year.astype(str),y=d.total_return*100,name=LABELS[n],marker_color=color))
    bars.update_layout(title='Yearly evaluation returns: 2026 is partial',yaxis_title='Total return (%)',template='plotly_white',height=460,legend=dict(orientation='h',y=-.2))
    q=quarterly[quarterly.model=='walk_forward'].copy()
    q['year']=q.quarter.str[:4]
    q['q']=q.quarter.str[-1].map(lambda x:'Q'+x)
    pivot=q.pivot(index='year',columns='q',values='total_return').reindex(columns=['Q1','Q2','Q3','Q4'])*100
    heat=go.Figure(go.Heatmap(z=pivot.values,x=pivot.columns,y=pivot.index,colorscale='RdBu',zmid=0,
                             text=pivot.round(1).values,texttemplate='%{text}%',hovertemplate='%{y} %{x}: %{z:.2f}%<extra></extra>'))
    heat.update_layout(title='Quarterly returns of annual-selection account; 2026 Q3 partial',height=550,template='plotly_white')
    parts=['<!doctype html><html><head><meta charset="utf-8"><title>Annual walk-forward evaluation</title>',
           '<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:18px;color:#24313a}p{line-height:1.6}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #ddd}td{text-align:right}</style></head><body>',
           '<h1>Annual walk-forward evaluation</h1><p>Five preceding calendar years select one of 21 fixed policies; the following year evaluates it. One continuous account carries holdings through policy changes, paying next-open reallocation costs. No resetting or splicing of annual portfolios. Quarterly results are slices of annual selections, not quarterly reselection.</p>',
           '<p><strong>Primary result: 2013–2023.</strong> Extended results include the previously viewed 2024–2026 period. First evaluation year is 2013; the GFC is in initial training. This is retrospective chronological evaluation: the candidate family and history have already been researched. It is not newly untouched evidence.</p>',
           '<p><a href="ANALYSIS.md">Interpretation</a> | <a href="protocol.json">Protocol and expectations</a> | <a href="schedule.csv">Annual selections</a> | <a href="annual.csv">Annual metrics</a> | <a href="quarterly.csv">Quarterly metrics</a></p>']
    for period in summary.period.unique():
        d=summary[summary.period==period].set_index('model')[['cagr','annual_vol','sharpe','max_drawdown','gross_turnover_per_year']].copy()
        d.index=d.index.map(LABELS)
        for col in ('cagr','annual_vol','max_drawdown'):d[col]=d[col].map(lambda v:f'{v:.2%}')
        for col in ('sharpe','gross_turnover_per_year'):d[col]=d[col].map(lambda v:f'{v:.3f}')
        parts.extend([f'<h2>{period.replace("_"," ")}</h2>',d.to_html()])
    for i,f in enumerate((fig,bars,heat)):parts.append(f.to_html(full_html=False,include_plotlyjs=True if i==0 else False))
    parts.extend(['<h2>Selection using only preceding data</h2>',schedule.to_html(index=False),
                  '<h2>Paired Sharpe uncertainty</h2><p>2,000 paired circular 20-session block resamples. Pointwise 95% intervals, no multiple-testing correction; these do not incorporate all uncertainty from rerunning model selection in alternate histories.</p>',paired.round(4).to_html(index=False),
                  '<p>Adjusted-price fractional holdings; lagged cash interest; 0.5% reserve; 1 bp commission and 5 bps slippage per side; no terminal liquidation. Static controls run continuously over identical dates. SPY/IEF 60/40 is a US Treasury benchmark, not a global or aggregate-bond benchmark.</p></body></html>'])
    (p/'report.html').write_text('\n'.join(parts),encoding='utf-8')


if __name__=='__main__':build(Path(__file__).resolve().parent/'outputs/rolling_evaluation')
