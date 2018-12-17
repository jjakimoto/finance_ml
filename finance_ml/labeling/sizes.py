import numpy as np
import pandas as pd
from scipy.stats import norm

from ..multiprocessing import mp_pandas_obj


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


def discrete_signal(signal, step_size):
    """Discretize signal
    
    Parameters
    ----------
    signal: pd.Series
        Signals for betting size ranged [-1, 1]
    step_size: float
        Discrete size
    
    Returns
    -------
    pd.Series
    """
    signal = (signal / step_size).round() * step_size
    # Cap
    signal[signal > 1] = 1
    signal[signal < -1] = -1
    return signal


def avg_active_signals(signals, num_threads=1, timestamps=None):
    """Average active signals

    Paramters
    ---------
    signals: pd.Series
    num_threads: 1
    timestamps: list, optional
        Timestamps used for output. When there is not active signal,
        value will be zero on that point. If not specified, use signals.index
    
    Return
    ------
    pd.Series
    """
    if timestamps is None:
        timestamps = set(signals['t1'].dropna().values)
        timestamps = list(timestamps.union(set(signals.index.values)))
        timestamps.sort()
    out = mp_pandas_obj(mp_avg_active_signals, ('molecule', timestamps), num_threads, signals=signals)
    return out


def mp_avg_active_signals(signals, molecule):
    """Function to calculate averaging with multiprocessing"""
    out = pd.Series()
    for loc in molecule:
        loc = pd.Timestamp(loc)
        cond = (signals.index <= loc) & ((loc < signals['t1']) | pd.isnull(signals['t1']))
        active_idx = signals[cond].index
        if len(active_idx) > 0:
            out[loc] = signals.loc[active_idx, 'signal'].mean()
        else:
            out[loc] = 0
    return out


def get_signal(events, prob, scale=1, step_size=None, num_classes=2, num_threads=1, **kwargs):
    """Return label

    Parameters
    ----------
    events: pd.DataFrame
        time: time of barrier
        type: type of barrier - tp, sl, or t1
        trgt: horizontal barrier width
        side: position side
    prob: pd.Series
        Probabilities signals
    scale: float
        Betting size scale
    step_size: float
        Discrete size
    num_classes: int, (default, 2)
        The number of classes
    num_threads: int, (default, 1)
        The number of threads used for averaging bets

    Returns
    -------
    pd.Series: bet size signal
    """
    # Get Signals
    if prob.shape[0] == 0:
        return pd.Series()
    signal = (prob - 1./num_classes) / (prob * (1 - prob))
    signal = pd.Series(2 * norm.cdf(signal.values) - 1, index=signal.index)
    if 'side' in events:
        signal = signal * events.loc[signal.index, 'side']
    if step_size is not None:
        signal = discrete_signal(signal, step_size=step_size)
    signal = scale * signal
    return signal