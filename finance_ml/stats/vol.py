import numpy as np
import pandas as pd


def get_daily_vol(close, span=100):
    """Estimate exponential average volatility"""
    use_idx = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    use_idx = use_idx[use_idx > 0]
    # Get rid of duplications in index
    use_idx = np.unique(use_idx)
    prev_idx = pd.Series(close.index[use_idx - 1], index=close.index[use_idx])
    ret = close.loc[prev_idx.index] / close.loc[prev_idx.values].values - 1
    vol = ret.ewm(span=span).std()
    return vol



