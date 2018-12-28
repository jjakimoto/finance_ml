import numpy as np
import pandas as pd

from ..multiprocessing import mp_pandas_obj


def mp_sample_weight(series, t1, num_co_events, molecule):
    weight = pd.Series(index=molecule)
    for t_in, t_out in t1.loc[weight.index].iteritems():
        weight.loc[t_in] = (
            series.loc[t_in:t_out] / num_co_events.loc[t_in:t_out]).sum()
    return weight.abs()


def get_sample_weight(series, t1, num_co_events, num_threads=1):
    """Calculate sampeling weight with considering some attributes
    
    Params
    ------
    series: pd.Series
        Used for assigning weight. Larger value, larger weight e.g., log return
    t1: pd.Series
    num_co_events: pd.Series
    num_threads: int
    
    Return
    ------
    pd.Series
    """
    weight = mp_pandas_obj(
        mp_sample_weight, ('molecule', t1.index),
        num_threads,
        series=series,
        t1=t1,
        num_co_events=num_co_events)
    return weight * weight.shape[0] / weight.sum()


def mp_uniq_weight(t1, num_co_events, molecule):
    """Calculate time sample weight utilizing occurence events information"""
    wght = pd.Series(index=molecule)
    for t_in, t_out in t1.loc[wght.index].iteritems():
        wght.loc[t_in] = (1. / num_co_events.loc[t_in:t_out]).mean()
    return wght


def get_uniq_weight(t1, num_co_events, num_threads=1):
    """Calculate time sample weight utilizing occurence events information
    
    Params
    ------
    t1: pd.Series
    num_co_events: pd.Series
        The number of co-occurence events
    num_threads: int
    
    Returns
    pd.Series
    """
    return mp_pandas_obj(
        mp_uniq_weight, ('molecule', t1.index),
        num_threads,
        t1=t1,
        num_co_events=num_co_events)
