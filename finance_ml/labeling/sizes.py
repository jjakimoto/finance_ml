import numpy as np
import pandas as pd


def get_sizes(close, events, sign_label=True):
    """Return bet sizes

    Parameters
    ----------
    close: pd.Series
    events: pd.DataFrame
        t1: time of barrier
        type: type of barrier - tp, sl, or t1
        trgt: horizontal barrier width
        side: position side
    sign_label: bool, (default True)
        If True, assign label for points touching vertical
        line accroing to return's sign

    Returns
    -------
    pd.Series: bet sizes
    """
    # Prices algined with events
    events = events.dropna(subset=['t1'])
    # All used indices
    time_idx = events.index.union(events['t1'].values).drop_duplicates()
    close = close.reindex(time_idx, method='bfill')
    # Create out object
    out = pd.DataFrame(index=events.index)
    out['ret'] = close.loc[events['t1'].values].values / close.loc[
        events.index] - 1.
    if 'side' in events:
        out['ret'] *= events['side']
        out['side'] = events['side']
    out['size'] = np.sign(out['ret'])
    if sign_label:
        out.loc[out['ret'] == 0, 'size'] = 1.
    else:
        # 0 when touching vertical line
        out['size'].loc[events['type'] == 't1'] = 0
    if 'side' in events:
        out.loc[out['ret'] <= 0, 'size'] = 0
    return out