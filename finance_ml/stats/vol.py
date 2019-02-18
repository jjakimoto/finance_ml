import numpy as np
import pandas as pd


def _get_ret(close, span=100, days=None, seconds=None):
    """Estimate exponential average volatility"""
    if days is None:
        delta = pd.Timedelta(seconds=seconds)
    else:
        delta = pd.Timedelta(days=days)
    use_idx = close.index.searchsorted(close.index - delta)
    prev_idx = pd.Series(use_idx, index=close.index)
    prev_idx = prev_idx[prev_idx > 0]
    # Get rid of duplications in index
    prev_idx = prev_idx.drop_duplicates()
    ret = close[prev_idx.index] / close[prev_idx].values - 1
    vol = ret.ewm(span=span).std()
    return vol


def get_vol(close, span=100, days=None, seconds=None):
    ret = _get_ret(close, span, days, seconds)
    vol = ret.ewm(span=span).std()
    return vol


def get_mean(close, span=100, days=None, seconds=None):
    ret = _get_ret(close, span, days, seconds)
    mean = ret.ewm(span=span).mean()
    return mean
