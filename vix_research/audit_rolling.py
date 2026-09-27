"""Post-run reconciliation of selection records and continuous return exports."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .rolling_evaluation import select_policy
from .literature_study import verify


def run():
    root=Path(__file__).resolve().parent
    p=root/'outputs/rolling_evaluation'
    schedule=pd.read_csv(p/'schedule.csv')
    history=pd.read_csv(p/'policy_history.csv',parse_dates=['date','train_end'])
    annual=pd.read_csv(p/'annual.csv')
    quarterly=pd.read_csv(p/'quarterly.csv')
    assert (history.train_end<history.date).all()
    assert len(schedule)==14
    for row in schedule.itertuples():
        training=pd.read_csv(p/f'training_{row.evaluation_year}.csv').set_index('model')
        assert len(training)==21 and select_policy(training)==row.selected
        assert pd.Timestamp(row.train_end)<pd.Timestamp(row.evaluation_start)
        h=history[history.date.dt.year==row.evaluation_year]
        assert (h.selected==row.selected).all()
    assert history.policy_changed.sum()==(history.selected!=history.selected.shift()).sum()
    max_error=0.
    for name in annual.model.unique():
        daily=pd.read_csv(p/'continuous'/name/'daily.csv',parse_dates=['date']).set_index('date')
        assert daily.index.equals(pd.DatetimeIndex(history.date))
        np.testing.assert_allclose(daily.nav,100000*(1+daily['return']).cumprod(),rtol=1e-12)
        for row in annual[annual.model==name].itertuples():
            quarters=quarterly[(quarterly.model==name)&quarterly.quarter.str.startswith(str(row.year))]
            delta=abs((1+quarters.total_return).prod()-1-row.total_return)
            max_error=max(max_error,delta)
            assert delta<1e-10
        cols=daily.filter(regex='^contribution_')
        assert np.max(abs(cols.sum(axis=1)-daily['return']))<1e-10
    html=(p/'report.html').read_text(encoding='utf-8')
    for f in ('ANALYSIS.md','protocol.json','schedule.csv','annual.csv','quarterly.csv'):
        assert f in html and (p/f).exists()
    verify(root/'outputs/literature',root/'data/market')
    result=dict(folds=14,candidates_per_fold=21,training_and_live_dates_disjoint=True,
                realised_policy_changes=int(history.policy_changed.sum()-1),
                quarterly_to_annual_max_error=max_error,continuous_nav_reconciliation=True,
                four_execution_tests_passed=True,report_links_checked=True)
    (p/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
