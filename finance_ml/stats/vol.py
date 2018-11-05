import numpy as np
import pandas as pd


def get_vol(close, span=100, days=None, seconds=None):
    """Estimate exponential average volatility"""
    if days is None:
        delta = pd.Timedelta(seconds=seconds)
    else:
        delta = pd.Timedelta(days=days)
    use_idx = close.index.searchsorted(close.index - delta)
    use_idx = use_idx[use_idx > 0]
    # Get rid of duplications in index
    use_idx = np.unique(use_idx)
    prev_idx = pd.Series(close.index[use_idx - 1], index=close.index[use_idx])
    ret = close[prev_idx.index] / close[prev_idx].values - 1
    vol = ret.ewm(span=span).std()
    return vol



