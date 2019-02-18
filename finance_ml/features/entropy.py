from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..multiprocessing import mp_pandas_obj


def plug_in(data, window):
    """Plug in Entropy Estimator
    
    Params
    ------
    data: list
    window: int
    
    Returns
    -------
    float: Estimated entropy
    dict: Probability mass function
    """
    pmf = calc_pmf(data, window)
    out = -sum([pmf[key] * np.log2(pmf[key]) for key in pmf.keys()])
    return out, pmf


def calc_pmf(data, window):
    """Calculate probability mass function
    
    Params
    ------
    data: list
    window: int
    
    Returns
    -------
    dict
    """
    lib = {}
    for i in range(window, len(data)):
        x = '_'.join([str(data_i) for data_i in data[i - window:i]])
        if x not in lib:
            lib[x] = [i - window]
        else:
            lib[x] += [
                i - window,
            ]
    num_samples = float(len(data) - window)
    pmf = {key: len(lib[key]) / num_samples for key in lib}
    return pmf


def lempel_zib_lib(data):
    """Calculate Lampel Ziv dictionary
    
    Params
    ------
    data: list
    
    Returns
    -------
    dict
    """
    i = 1
    lib = [str(data[0])]
    while i < len(data):
        for j in range(i, len(data)):
            x = '_'.join([str(data_i) for data_i in data[i:j + 1]])
            if x not in lib:
                lib.append(x)
                break
        i = j + 1
    return lib


def match_length(data, i, n):
    """Calculate math length
    
    Params
    ------
    data: list
    i: int, start point
    n: int, window size
    
    Returns
    -------
    int: length of the longest matched substring + 1
    str: the longest mathed substring
    """
    sub_str = ''
    for l in range(n):
        msg1 = '_'.join([str(data_i) for data_i in data[i:i + l + 1]])
        for j in range(max(i - n, 0), i):
            msg0 = '_'.join([str(data_i) for data_i in data[j:j + l + 1]])
            if msg1 == msg0:
                sub_str = msg1
                break
    return len(sub_str.split('_')) + 1, sub_str


def konto(data, window=None, verbose=0):
    """Calculate Kontonyiasnnis' LZ entropy estimate
    
    Params
    ------
    data: list
    window: int, optional
    verbose: int, default 0
        If 1, show the progress bar
    """
    out = {'num': 0, 'sum': 0, 'sub_str': []}
    if window is None:
        points = range(1, len(data) // 2 + 1)
    else:
        window = min(window, len(data) // 2)
        poitns = range(window, len(data) - window + 1)
    if verbose == 1:
        points = tqdm(points)
    for i in points:
        if window is None:
            l, msg = match_length(data, i, i)
            out['sum'] += np.log2(i + 1) / l
        else:
            l, msg = match_length(data, i, window)
            out['sum'] += np.log(i + 1) / l
        out['sub_str'].append(msg)
        out['num'] += 1
    out['h'] = out['sum'] / out['num']
    out['r'] = 1 - out['h'] / np.log2(len(data))
    return out


def mp_get_entropy_rate(series, lag, molecule):
    delta = timedelta(seconds=lag)
    entropy = pd.Series(index=molecule)
    for t in molecule:
        series_ = series[t - delta:t]
        entropy_t = konto(series_.values, verbose=0)
        entropy.loc[t] = entropy_t['h']
    return entropy


def get_entropy_rate(series, lag, num_threads=1):
    """Calculate entropy rate for time series

    Params
    ------
    series: pd.Series
    lag: int
        Time slide length (seconds)
    num_threads: int, default 1
    
    Returns
    -------
    pd.Series
    """
    start = series.index[0] + timedelta(seconds=lag)
    return mp_pandas_obj(
        mp_get_entropy_rate, ('molecule', series[start:].index),
        num_threads,
        series=series,
        lag=lag)
