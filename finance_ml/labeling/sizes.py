import numpy as np
import pandas as pd


def get_sizes(close, events, min_ret=0, sign_label=True, zero_label=0):
    """Return label

    Parameters
    ----------
    close: pd.Series
    events: pd.DataFrame
        time: time of barrier
        type: type of barrier - tp, sl, or t1
        trgt: horizontal barrier width
        side: position side
    min_ret: float
        Minimum of absolute value for labeling non zero label. min_ret >=0
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
    out.loc[(out['ret'] <= min_ret) & (out['ret'] >= -min_ret), 'size'] = zero_label
    if 'side' in events:
        out.loc[out['ret'] <= min_ret, 'size'] = zero_label
    if not sign_label:
        out['size'].loc[events['type'] == 't1'] = zero_label
    return out
