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


def get_ind_matrix(timestamps, t1, num_threads=1):
    return mp_pandas_obj(
        mp_ind_matrix, ('molecule', t1.index),
        num_threads,
        timestamps=timestamps,
        t1=t1)


def mp_ind_matrix(timestamps, t1, molecule):
    t1 = t1.loc[molecule]
    ind_matrix = pd.DataFrame(0, index=timestamps, columns=molecule)
    for i, (t0, t1) in enumerate(t1.iteritems()):
        ind_matrix.loc[t0:t1] = 1
    return ind_matrix