import pandas as pd


def get_ind_matrix(bar_idx, t1):
    ind_m = pd.DataFrame(0, index=bar_idx,
                         columns=range(t1.shape[0]))
    for i, (t0_, t1_) in enumerate(t1.iteritems()):
        ind_m.loc[t0_:t1_, i] = 1
    return ind_m


def get_avg_uniq(ind_m, c=None):
    if c is None:
        c = ind_m.sum(axis=1)
    ind_m = ind_m.loc[c > 0]
    c = c.loc[c > 0]
    u = ind_m.div(c, axis=0)
    avg_u = u[u > 0].mean()
    avg_u = avg_u.fillna(0)
    return avg_u
