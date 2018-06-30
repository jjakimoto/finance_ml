import numpy as np
import pandas as pd


def get_bins(events, close):
    # Prices algined with events
    events = events.dropna(subset=['t1'])
    px = events.index.union(events['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # Create out object
    out = pd.DataFrame(index=events.index)
    out['ret'] = px.loc[events['t1'].values].values / px.loc[events.index] - 1.
    if 'side' in events:
        out['ret'] *= events['side']
    out['bin'] = np.sign(out['ret'])
    # 0 when touching vertical line
    out['bin'].loc[events['t1_type'] == 't1'] = 0
    if 'side' in events:
        out.loc[out['ret'] <= 0, 'bin'] = 0
    return out