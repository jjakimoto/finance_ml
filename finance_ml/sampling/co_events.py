import numpy as np
import pandas as pd

from ..multiprocessing import mp_pandas_obj


def mp_num_co_events(timestamps, t1, molecule):
    """Calculate the number of co events for multiprocessing"""
    # Find events that span the period defined by molecule
    t1 = t1.fillna(timestamps[-1])
    t1 = t1[t1 >= molecule[0]]
    t1 = t1.loc[:t1[molecule].max()]
    # Count the events
    iloc = timestamps.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=timestamps[iloc[0]:iloc[1] + 1])
    for t_in, t_out in t1.iteritems():
        count.loc[t_in:t_out] += 1
    return count.loc[molecule[0]:t1[molecule].max()]


def get_num_co_events(timestamps, t1, num_threads=1):
    """Calculate the number of co events
    
    Params
    ------
    timestamps: DatetimeIndex
        The timesstamps defining the range of searching
    t1: pd.Series
    num_threads: int
    
    Returns
    pd.Series: each value corresponds to the number of co occurence
    """
    return mp_pandas_obj(
        mp_num_co_events, ('molecule', t1.index),
        num_threads,
        timestamps=timestamps,
        t1=t1)
