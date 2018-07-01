import numpy as np
from .utils import get_avg_uniq


def seq_bootstrap(ind_m, s_length=None):
    if s_length is None:
        s_length = ind_m.shape[1]
    phi = []
    while len(phi) < s_length:
        c = ind_m[phi].sum(axis=1) + 1
        avg_u = get_avg_uniq(ind_m, c)
        prob = (avg_u / avg_u.sum()).values
        phi += [np.random.choice(ind_m.columns, p=prob)]
    return phi