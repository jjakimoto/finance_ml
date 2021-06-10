import numpy as np
import pandas as pd

from .clustering import cluster_kmeans_base
from .utils import cov2corr


def opt_portfolio(cov, mu=None):
    inv = np.linalg.inv(cov)
    ones = np.ones(shape=(inv.shape[0], 1))
    if mu is None:
        mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)
    return w

def opt_portfolio_nco(cov, mu=None, max_num_clusters=None, min_num_clusters=2, n_init=10):
    cov = pd.DataFrame(cov)
    if mu is not None:
        mu = pd.Series(mu[:, 0])
    corr1 = cov2corr(cov)
    if max_num_clusters is None:
        max_num_clusters = int(corr1.shape[0] / 2)
    corr1, clstrs, _ = cluster_kmeans_base(corr1, max_num_clusters, min_num_clusters, n_init)
    w_intra = pd.DataFrame(0, index=cov.index, columns=clstrs.keys())
    for key in clstrs.keys():
        cov_ = cov.loc[clstrs[key], clstrs[key]].values
        if mu is None:
            mu_ = None
        else:
            mu_ = mu.loc[clstrs[key]].values.reshape(-1, 1)
        # Compute optimal portfolio within cluster
        w_intra.loc[clstrs[key], key] = opt_portfolio(cov_, mu_).flatten()
    # n_clusters x n_clusters covariance
    inter_cov = w_intra.T.dot(np.dot(cov, w_intra))
    if mu is None:
        inter_mu = None
    else:
        inter_mu= w_intra.T.dot(mu)
    # Calculate cluster allocation
    w_inter = pd.Series(opt_portfolio(inter_cov, inter_mu).flatten(), index=inter_cov.index)
    nco = w_intra.mul(w_inter, axis=1).sum(axis=1).values.reshape(-1, 1)
    return nco