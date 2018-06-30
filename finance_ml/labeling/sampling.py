import numbers
import pandas as pd


def cusum_filter(close, h):
    # asssum that E y_t = y_{t-1}
    t_events = []
    s_pos, s_neg = 0, 0
    ret = close.pct_change().dropna()
    diff = ret.diff().dropna()
    # time variant threshold
    if isinstance(h, numbers.Number):
        h = pd.Series(h, index=diff.index)
    h = h.reindex(diff.index, method='bfill')
    h = h.dropna()
    for t in h.index:
        s_pos = max(0, s_pos + diff.loc[t])
        s_neg = min(0, s_neg + diff.loc[t])
        if s_pos > h.loc[t]:
            s_pos = 0
            t_events.append(t)
        elif s_neg < -h.loc[t]:
            s_neg = 0
            t_events.append(t)
    return pd.DatetimeIndex(t_events)