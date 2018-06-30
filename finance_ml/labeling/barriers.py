import pandas as pd

from ..multiprocessing import mp_pandas_obj


def apply_ptslt1(close, events, ptsl, molecule):
    """Return datafram about if price touches the boundary"""
    # Sample a subset with specific indices
    _events = events.loc[molecule]
    out = pd.DataFrame(index=_events.index)
    # Set Profit Taking and Stop Loss
    if ptsl[0] > 0:
        pt = ptsl[0] * _events["trgt"]
    else:
        # Switch off profit taking
        pt = pd.Series(index=_events.index)
    if ptsl[1] > 0:
        sl = -ptsl[1] * _events["trgt"]
    else:
        # Switch off stop loss
        sl = pd.Series(index=_events.index)
    # Replace undefined value with the last time index
    time_limits = _events["t1"].fillna(close.index[-1])
    for loc, t1 in time_limits.iteritems():
        df = close[loc:t1]
        # Change the direction depending on the side
        df = (df / close[loc] - 1) * _events.at[loc, 'side']
        out.at[loc, 'sl'] = df[df < sl[loc]].index.min()
        out.at[loc, 'pt'] = df[df > pt[loc]].index.min()
    out['t1'] = _events['t1'].copy(deep=True)
    return out


def get_events(close, t_events, ptsl, trgt, min_ret=0, num_threads=1,
                  t1=False, side=None):
    # Get sampled target values
    trgt = trgt.loc[t_events]
    trgt = trgt[trgt > min_ret]
    # Get time boundary t1
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=t_events)
    # Define the side
    if side is None:
        _side = pd.Series(1., index=trgt.index)
        _ptsl = [ptsl, ptsl]
    else:
        _side = side.loc[trgt.index]
        _ptsl = ptsl[:2]
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': _side}, axis=1)
    events = events.dropna(subset=['trgt'])
    time_idx = mp_pandas_obj(func=apply_ptslt1,
                             pd_obj=('molecule', events.index),
                             num_threads=num_threads,
                             close=close, events=events, ptsl=_ptsl)
    # Skip when all of barrier are not touched
    time_idx = time_idx.dropna(how='all')
    events['t1_type'] = time_idx.idxmin(axis=1)
    events['t1'] = time_idx.min(axis=1)
    if side is None:
        events = events.drop('side', axis=1)
    return events


def get_t1(close, t_events, num_days):
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[:t1.shape[0]])
    return t1
