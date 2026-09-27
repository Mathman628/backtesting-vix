"""Fetch only additional ETF histories; preserve frozen original market inputs."""
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import yfinance as yf


def run():
    root=Path(__file__).resolve().parent
    folder=root/'data/research_extension'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'prices.csv').exists():
        print('Extension data already cached; no replacement.')
        return
    yf.set_tz_cache_location(str(folder/'yfinance_cache'))
    records=[]
    for symbol in ('EFA','VNQ','GSG'):
        frame=yf.download(symbol,start='2006-01-01',end='2026-09-24',auto_adjust=True,
                          progress=False,threads=False,multi_level_index=False)
        if frame.empty:raise ValueError(f'No data for {symbol}')
        for date,row in frame.iterrows():
            records.append(dict(date=pd.Timestamp(date).strftime('%Y-%m-%d'),symbol=symbol,
                                adjusted_open=row['Open'],adjusted_close=row['Close']))
        print(symbol,len(frame),frame.index[0],frame.index[-1],flush=True)
    pd.DataFrame(records).to_csv(folder/'prices.csv',index=False)
    (folder/'manifest.json').write_text(json.dumps(dict(downloaded_utc=datetime.now(timezone.utc).isoformat(),
        source='Yahoo Finance via yfinance',auto_adjust=True,start='2006-01-01',end_exclusive='2026-09-24',
        symbols=['EFA','VNQ','GSG'],original_data_unchanged=True),indent=2),encoding='utf-8')


if __name__=='__main__':run()
