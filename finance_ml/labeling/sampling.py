import numbers
import pandas as pd


def cusum_filter(close, h):
    """Sample points with CUSUM Filter

    Parameters
    ----------
    close: pd.Series
    h: pd.Series
        Threasholds to sampmle points

    Returns
    -------
    pd.DatetimeIndex: Sampled data points
    """
    # asssum that E y_t = y_{t-1}
    s_pos, s_neg = 0, 0
    diff = close.diff().dropna()
    # time variant threshold
    if isinstance(h, numbers.Number):
        h = pd.Series(h, index=diff.index)
    h = h.reindex(diff.index, method='bfill')
    h = h.dropna()
    timestamps = []
    th = h.loc[h.index[0]]
    for t in h.index:
        s_pos = max(0, s_pos + diff.loc[t])
        s_neg = min(0, s_neg + diff.loc[t])
        if s_pos > th:
            s_pos = 0
            timestamps.append(t)
            th = h.loc[t]
        elif s_neg < -th:
            s_neg = 0
            timestamps.append(t)
            th = h.loc[t]
    return pd.DatetimeIndex(timestamps)