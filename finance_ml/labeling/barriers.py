import pandas as pd

from ..multiprocessing import mp_pandas_obj
from ..constants import LONG, SHORT
from .sizes import get_sizes


def get_touch_idx(close, events, sltp, molecule=None):
    """Return timestamps of when data points touch the barriers

    Parameters
    ----------
    close: pd.Series
        Close price series
    events: pd.DataFrame with columns: 't1', 'trgt', and 'side'
        t1: time stamp of vertical barrier, could be np.nan
        trgt: unit of width of horizontal barriers
        side: Side label for metalabeling
    sltp: list
        Coefficients of width of Stop Loss and Take Profit.
        sltp[0] and sltp[1] correspond to width of stop loss
        and take profit, respectively. If 0 or negative, the barrier
        is siwthced off.
    molecule: list, optional
        Subset of indices of events to be processed

    Returns
    -------
    pd.DataFrame: each colum corresponds to the time to touch the barrier
    """
    # Sample a subset with specific indices
    if molecule is not None:
        _events = events.loc[molecule]
    else:
        _events = events
    touch_idx = pd.DataFrame(index=_events.index)
    # Set Stop Loss and Take Profoit
    if sltp[0] > 0:
        sls = -sltp[0] * _events["trgt"]
    else:
        # Switch off stop loss
        sls = pd.Series(index=_events.index)
    if sltp[1] > 0:
        tps = sltp[1] * _events["trgt"]
    else:
        # Switch off profit taking
        tps = pd.Series(index=_events.index)
    # Replace undefined value with the last time index
    vertical_lines = _events["t1"].fillna(close.index[-1])
    for loc, t1 in vertical_lines.iteritems():
        df = close[loc:t1]
        # Change the direction depending on the side
        df = (df / close[loc] - 1) * _events.at[loc, 'side']
        touch_idx.at[loc, 'sl'] = df[df < sls[loc]].index.min()
        touch_idx.at[loc, 'tp'] = df[df > tps[loc]].index.min()
    touch_idx['t1'] = _events['t1'].copy(deep=True)
    return touch_idx


def get_events(close, timestamps, sltp, trgt, min_ret=0,
               num_threads=1, t1=None, side=None):
    """Return DataFrame containing infomation defining barriers

    Parameters
    ----------
    close: pd.Series
        Close price series
    timestamps: pd.DatetimeIndex
        sampled points to analyze
    sltp: list
        Coefficients of width of Stop Loss and Take Profit.
        sltp[0] and sltp[1] correspond to width of stop loss
        and take profit, respectively. If 0 or negative, the barrier
        is siwthced off.
    trgt: pd.Series
        Time series of threashold
    min_ret: float, (default 0)
        Minimum value of points to label
    num_threads: int, (default 1)
        The number of threads to use
    t1: pd.Series, optional
        Vertical lines
    side: pd.Series, optional
        Side of trading positions

    Returns
    -------
    pd.DataFrame with columns: 't1', 'trgt', 'type', and 'side'
    """
    # Get sampled target values
    trgt = trgt.loc[timestamps]
    trgt = trgt[trgt > min_ret]
    if len(trgt) == 0:
        return pd.DataFrame(columns=['t1', 'trgt', 'side'])
    # Get time boundary t1
    if t1 is None:
        t1 = pd.Series(pd.NaT, index=timestamps)
    # slpt has to be either of integer, list or tuple
    if isinstance(sltp, list) or isinstance(sltp, tuple):
        _sltp = sltp[:2]
    else:
        _sltp = [sltp, sltp]
    # Define the side
    if side is None:
        # Default is LONG
        _side = pd.Series(LONG, index=trgt.index)
    else:
        _side = side.loc[trgt.index]
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': _side}, axis=1)
    events = events.dropna(subset=['trgt'])
    time_idx = mp_pandas_obj(func=get_touch_idx,
                             pd_obj=('molecule', events.index),
                             num_threads=num_threads,
                             close=close, events=events, sltp=_sltp)
    # Skip when all of barrier are not touched
    time_idx = time_idx.dropna(how='all')
    events['type'] = time_idx.idxmin(axis=1)
    events['t1'] = time_idx.min(axis=1)
    if side is None:
        events = events.drop('side', axis=1)
    return events


def get_t1(close, timestamps, num_days):
    """Return horizontal timestamps

    Note
    ----
    Not include the case to hit the vertical line at the end of close.index

    Parameters
    ----------
    close: pd.Series
    timestamps: pd.DatetimeIndex
    num_days: int
        The number of forward dates for vertical barrier

    Returns
    -------
    pd.Series: Vertical barrier timestamps
    """
    t1 = close.index.searchsorted(timestamps + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=timestamps[:t1.shape[0]])
    return t1


def get_barrier_labels(close, timestamps, trgt, sltp=[1, 1],
                       num_days=1, min_ret=0, num_threads=16,
                       side=None, sign_label=True):
    """Return Labels for triple barriesr

    Parameters
    ----------
    close: pd.Series
    timestamps: pd.DatetimeIndex
        sampled points to analyze
    trgt: pd.Series
        Time series of threshold
    sltp: list, (default [1, 1]
        Coefficients of width of Stop Loss and Take Profit.
        sltp[0] and sltp[1] correspond to width of stop loss
        and take profit, respectively. If 0 or negative, the barrier
        is switched off.
    num_days: int, (default, 1)
        The number of forward dates for vertical barrier
    min_ret: float, (default 0)
        Minimum value of points to label
    num_threads: int, (default 16)
        The number of threads to use
    side: pd.Series, optional
        Side of trading positions
    sign_label: bool, (default True)
        If True, assign label for points touching vertical
        line according to return's sign

    Returns
    -------
    pd.Series: label
    """
    t1 = get_t1(close, timestamps, num_days)
    events = get_events(close, timestamps,
                        sltp=sltp,
                        trgt=trgt,
                        min_ret=min_ret,
                        num_threads=num_threads,
                        t1=t1, side=side)
    sizes = get_sizes(close, events, sign_label=sign_label)
    return sizes['size']
