import numpy as np
import pandas as pd


def get_sizes(close, events, sign_label=True, zero_label=None):
    """Return label

    Parameters
    ----------
    close: pd.Series
    events: pd.DataFrame
        time: time of barrier
        type: type of barrier - tp, sl, or t1
        trgt: horizontal barrier width
        side: position side
    sign_label: bool, (default True)
        If True, assign label for points touching vertical
        line accroing to return's sign
    zero_label: int, optional
        If specified, use it for the label of zero value of return
        If not, get rid of samples

    Returns
    -------
    pd.Series: bet sizes
    """
    # Prices algined with events
    events = events.dropna(subset=['time'])
    # All used indices
    time_idx = events.index.union(events['time'].values).drop_duplicates()
    close = close.reindex(time_idx, method='bfill')
    # Create out object
    out = pd.DataFrame(index=events.index)
    out['ret'] = close.loc[events['time'].values].values / close.loc[
        events.index] - 1.
    # Modify return according to the side
    if 'side' in events:
        out['ret'] *= events['side']
        out['side'] = events['side']
    # Assign labels
    out['size'] = np.sign(out['ret'])
    if zero_label is None:
        out = out.loc[out['ret'] != 0]
    else:
        out.loc[out['ret'] ==0, 'size'] = zero_label
    if 'side' in events:
        out.loc[out['ret'] <= 0, 'size'] = 0
    if not sign_label:
        out['size'].loc[events['type'] == 't1'] = 0
    return out
