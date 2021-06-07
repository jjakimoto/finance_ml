import numbers
import pandas as pd
import numpy as np
import multiprocessing as mp

from ..multiprocessing import mp_pandas_obj


def _cusum_side(diff, h, k=0, molecule=None):
    side = []
    s_pos, s_neg = 0, 0
    timestamps = []
    th = None
    for t in molecule:
        if th is None:
            th = h.loc[t]
        s_pos = max(0, s_pos + diff.loc[t] - k)
        s_neg = min(0, s_neg + diff.loc[t] + k)
        if s_pos > th:
            s_pos = 0
            timestamps.append(t)
            th = h.loc[t]
            side.append(1)
        elif s_neg < -th:
            s_neg = 0
            timestamps.append(t)
            th = h.loc[t]
            side.append(-1)
    side = pd.Series(side, index=pd.DatetimeIndex(timestamps))
    return side


def cusum_side(close, h, k=0, use_log=True, num_threads=None):
    """Sample points with CUSUM Filter and use its direction as betting side

    Args:
        close (pd.Series): Price series

        h (float or pd.Series): Threasholds to sampmle points.\
            If specified with float, translate to pd.Series(h, index=close.index)

        k (float, optional): Minimum speed parameter to hit threashold.\
            Defaults to 0, which means inactive

    Returns:
        pd.Series: Betting sides at sampled points
    """
    if num_threads is None:
        num_threads = mp.cpu_count()
    # asssum that E y_t = y_{t-1}
    side = []
    s_pos, s_neg = 0, 0
    if use_log:
        diff = np.log(close).diff().dropna()
    else:
        diff = close.diff().dropna()
    # time variant threshold
    if isinstance(h, numbers.Number):
        h = pd.Series(h, index=diff.index)
    h = h.reindex(diff.index, method='bfill')
    h = h.dropna()
    side = mp_pandas_obj(func=_cusum_side,
                         pd_obj=('molecule', h.index),
                         num_threads=num_threads,
                         diff=diff, h=h, k=k)
    return side