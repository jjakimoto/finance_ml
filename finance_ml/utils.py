import numbers
from copy import deepcopy

import numpy as np


def sign_log(x, scale=1):
    const = 1
    if isinstance(x, numbers.Number):
        if x >= 0:
            return np.log(const + scale * x)
        else:
            return np.log(const + scale * np.abs(x))
    x = deepcopy(x)
    x[x >= 0] = np.log(const + scale * np.abs(x[x >= 0]))
    x[x < 0] = -np.log(const + scale * np.abs(x[x < 0]))
    return x

def cov2corr(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1] = -1
    corr[corr > 1] = 1
    return corr

def corr2cov(corr, std):
    return corr * np.outer(std, std)
