import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm, tqdm_notebook


def get_weights_FFD(d, thres, max_size=10000):
    """Get coefficient for calculating fractional derivative
    
    Params
    ------
    d: int
    thres: float
    max_size: int, defaut 1e4
        Set the maximum size for stability
        
    Returns
    -------
    array-like
    """
    w = [1.]
    for k in range(1, max_size):
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) <= thres:
            break
        w.append(w_)
    w = np.array(w)
    return w


def frac_diff_FFD(series, d, lag=1, thres=1e-5, max_size=10000):
    """Compute Fractional Differentiation
    
    Params
    ------
    series: pd.Series
    d: float, the degree of differentiation
    lag: int, default 1
        The lag scale when making differential like series.diff(lag)
    thres: float, default 1e-5
        Threshold to determine fixed length window
    
    Returns
    -------
    pd.Series
    """
    max_size = int(max_size / lag)
    w = get_weights_FFD(d, thres, max_size)
    width = len(w)
    series_ = series.fillna(method='ffill').dropna()
    rolling_array = []
    for i in range(width):
        rolling_array.append(series_.shift(i * lag).values)
    rolling_array = np.array(rolling_array)
    series_val = np.dot(rolling_array.T, w)
    series = pd.Series(index=series.index)
    timestamps = series.index[-len(series_val):]
    series.loc[timestamps] = series_val
    return series


def get_opt_d(series, ds=None, lag=1, thres=1e-5, max_size=10000,
              p_thres=1e-2, autolag=None, verbose=1, **kwargs):
    """Find minimum value of degree of stationary differntial
    
    Params
    ------
    series: pd.Series
    ds: array-like, default np.linspace(0, 1, 100)
        Search space of degree.
    lag: int, default 1
        The lag scale when making differential like series.diff(lag)
    thres: float, default 1e-5
        Threshold to determine fixed length window
    p_threds: float, default 1e-2
    auto_lag: str, optional
    verbose: int, default 1
        If 1 or 2, show the progress bar. 2 for notebook
    kwargs: paramters for ADF
    
    Returns
    -------
    int, optimal degree
    """
    if ds is None:
        ds = np.linspace(0, 1, 100)
    # Sort to ascending order
    ds = np.array(ds)
    sort_idx = np.argsort(ds)
    ds = ds[sort_idx]
    if verbose == 2:
        iter_ds = tqdm_notebook(ds)
    elif verbose == 1:
        iter_ds = tqdm(ds)
    else:
        iter_ds = ds
    opt_d = ds[-1]
    # Compute pval for each d
    for d in iter_ds:
        diff = frac_diff_FFD(series, d=d, thres=thres, max_size=max_size)
        pval = adfuller(diff.dropna().values, autolag=autolag, **kwargs)[1]
        if pval < p_thres:
            opt_d = d
            break
    return opt_d