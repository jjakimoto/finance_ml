import numbers
import numpy as np
import pandas as pd
from scipy.stats import norm, t

from ..multiprocessing import mp_pandas_obj


# Specific Betting Size Calculation
###############################################################
def get_gaussian_betsize(probs, num_classes=2, eps=1e-4):
    """Translate probability to bettingsize

    Args:
        probs (array-like)
        num_classes (int, optional): Defaults to 2

    Returns:
        array-like: Signals after gaussian transform
    """
    max_prob = 1 - eps
    min_prob = eps
    if isinstance(probs, numbers.Number):
        if probs >= min_prob and probs <= max_prob:
            signal = (probs - 1. / num_classes) / np.sqrt(probs * (1 - probs))
            signal = 2 * norm.cdf(signal) - 1
        elif probs < min_prob:
            signal = -1
        elif probs > max_prob:
            signal = 1
        else:
            raise ValueError(f"Unkonwn probabilty: {probs}")
    else:
        signal = probs.copy()
        signal[probs >= max_prob] = 1
        signal[probs <= min_prob] = -1
        cond = (probs < max_prob) & (probs > min_prob)
        signal[cond] = (probs[cond] - 1. / num_classes) / np.sqrt(probs[cond] * (1 - probs[cond]))
        signal[cond] = 2 * norm.cdf(signal[cond]) - 1
    return signal


def get_tstats_betsize(probs, N, num_classes=2, eps=1e-4):
    """Translate probability to bettingsize

    Args:
        probs (array-like)
        N (int): The number of estimators used for generating probs
        num_classes (int, optional): Defaults to 2

    Returns:
        array-like: Signals after gaussian transform
    """
    max_prob = 1 - eps
    min_prob = eps
    if isinstance(probs, numbers.Number):
        if probs >= min_prob and probs <= max_prob:
            signal = (probs - 1. / num_classes) / np.sqrt(probs * (1 - probs)) * np.sqrt(N)
            signal = 2 * t.cdf(signal, df=N-1) - 1
        elif probs < min_prob:
            signal = -1
        elif probs > max_prob:
            signal = 1
        else:
            raise ValueError(f"Unkonwn probabilty: {probs}")
    else:
        signal = probs.copy()
        signal[probs >= max_prob] = 1
        signal[probs <= min_prob] = -1
        cond = (probs < max_prob) & (probs > min_prob)
        signal[cond] = (probs[cond] - 1. / num_classes) / np.sqrt(probs[cond] * (1 - probs[cond])) * np.sqrt(N)
        signal[cond] = 2 * t.cdf(signal[cond], df=N-1) - 1
    return signal


# Aggregate Signals
#####################################################################
def discrete_signals(signals, step_size):
    """Discretize signals
    
    Args:
        signals (pd.Series or float): Signals for betting size ranged [-1, 1]
        
        step_size (float): Discrete size ranged [0, 1]
    
    Returns:
        pd.Series or float: Discretized signals. If signals is pd.Series,\
            return value is pd.Series. If signals is float, return value\
            is float
    """
    if isinstance(signals, numbers.Number):
        signals = round(signals / step_size) * step_size
        signals = min(1, signals)
        signals = max(-1, signals)
    else:
        signals = (signals / step_size).round() * step_size
        signals[signals > 1] = 1
        signals[signals < -1] = -1
    return signals


def avg_active_signals(signals, num_threads=1, timestamps=None):
    """Average active signals

    Args:
        signals (pd.DataFrame): With keys: 't1' and 'signal'
            - t1, signal effective time boundary.
            - signal, signal value

        num_threads (int, optional): The number of processor used for calculation.\
            Defaults to 1.

        timestamps (list, optional): Timestamps used for output. When there is no active signal,\
            value will be zero on that point. If not specified, use signals.index.
    
    Returns:
        pd.Series: Averaged signals
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


# Signal Translation
#################################################################################
def get_betsize(probs,
                events=None,
                scale=1,
                step_size=None,
                signal_func=None,
                num_classes=2,
                num_threads=1,
                **kwargs):
    """Average and discretize signals from probability

    Args:
        events (pd.DataFrame): With the following keys
            - time, time of barrier
            - type, type of barrier - tp, sl, or t1
            - trgt, horizontal barrier width
            - side, position side

        probs (pd.Series): Probability signals

        scale (float): Betting size scale

        step_size (float, optional): If specified, discretize signals.\
            The value is ranged [0, 1]

        num_classes (int, optional): The number of classes. Defaults to 2.

        num_threads (int, optional): The number of threads used for averaging bets.\
            Defaults to 1.

    Returns:
        pd.Series: bet size signals
    """
    # Get Signals
    if probs.shape[0] == 0:
        return pd.Series()
    if signal_func is None:
        signal_func = get_gaussian_betsize
    signal = pd.Series(signal_func(probs, num_classes=num_classes, **kwargs), index=probs.index)
    if events and 'side' in events:
        signal = signal * events.loc[signal.index, 'side']
    if step_size is not None:
        signal = discrete_signals(signal, step_size=step_size)
    signal = scale * signal
    return signal