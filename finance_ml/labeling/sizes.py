import numbers
import numpy as np
import pandas as pd
from scipy.stats import norm

from ..multiprocessing import mp_pandas_obj
from .utils import get_gaussian_betsize


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
    if isinstance(signal, numbers.Number):
        signal = round(signal / step_size) * step_size
        signal = min(1, signal)
        signal = max(-1, signal)
    else:
        signal = (signal / step_size).round() * step_size
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
    out = mp_pandas_obj(
        mp_avg_active_signals, ('molecule', timestamps),
        num_threads,
        signals=signals)
    return out


def mp_avg_active_signals(signals, molecule):
    """Function to calculate averaging with multiprocessing"""
    out = pd.Series()
    for loc in molecule:
        loc = pd.Timestamp(loc)
        cond = (signals.index <= loc) & (
            (loc < signals['t1']) | pd.isnull(signals['t1']))
        active_idx = signals[cond].index
        if len(active_idx) > 0:
            out[loc] = signals.loc[active_idx, 'signal'].mean()
        else:
            out[loc] = 0
    return out


def get_signal(prob,
               events=None,
               scale=1,
               step_size=None,
               num_classes=2,
               num_threads=1,
               **kwargs):
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
    signal = (prob - 1. / num_classes) / (prob * (1 - prob))
    signal = pd.Series(
        get_gaussian_betsize(prob, num_classes), index=prob.index)
    if events and 'side' in events:
        signal = signal * events.loc[signal.index, 'side']
    if step_size is not None:
        signal = discrete_signal(signal, step_size=step_size)
    signal = scale * signal
    return signal