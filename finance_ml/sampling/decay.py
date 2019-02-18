import numpy as np
import pandas as pd


def get_time_decay(uniq_weight, last=1.):
    """Calculate time decay weight
    
    Params
    ------
    uniq_weight: pd.Series
        Sampling weight calculated label uniqueness
    last: float, default 1, no decay
        Parameter to detemine the slope and constant
    
    Returns
    -------
    pd.Series
    """
    weight = uniq_weight.sort_index().cumsum()
    if last > 0:
        slope = (1 - last) / weight.iloc[-1]
    else:
        slope = 1 / ((1 + last) * weight.iloc[-1])
    const = 1. - slope * weight.iloc[-1]
    weight = const + slope * weight
    weight[weight < 0] = 0
    return weight