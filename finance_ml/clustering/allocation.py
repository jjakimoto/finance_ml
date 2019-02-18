import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from .metrics import get_corr_dist
from .quasi import get_quasi_diag


def get_rec_bipart(cov, sort_idx):
    """Compute portfolio weight by recursive bisection
    
    Params
    ------
    cov: pd.DataFrame
    sort_idx: pd.Series
        Sorted index by quasi diagonalization
    
    Returns
    -------
    pd.Series
    """
    weight = pd.Series(1, index=sort_idx)
    # Initialize all in one cluster
    cl_items = [sort_idx]
    while len(cl_items) > 0:
        cl_items_ = []
        for cl in cl_items:
            # Split into half for each cluter
            if len(cl) >= 2:
                cl_items_.append(cl[0:len(cl) // 2])
                cl_items_.append(cl[len(cl) // 2:len(cl)])
        # Update cluster
        cl_items = cl_items_
        for i in range(0, len(cl_items), 2):
            cl0 = cl_items[i]
            cl1 = cl_items[i + 1]
            var0 = get_cluster_var(cov, cl0)
            var1 = get_cluster_var(cov, cl1)
            alpha = var1 / (var0 + var1)
            weight[cl0] *= alpha
            weight[cl1] *= 1 - alpha
    return weight


def get_ivp(cov):
    """Compute inverse variance portfolio
    
    Params
    ------
    cov: pd.DataFrame
    
    Returns
    -------
    np.array
    """
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def get_cluster_var(cov, cl_items):
    """Compute variance per cluster
    
    Params
    ------
    cov: pd.DataFrame
    cl_items: pd.Series
    
    Returns
    -------
    float
    """
    cov_cl = cov.loc[cl_items, cl_items]
    w = get_ivp(cov_cl).reshape(-1, 1)
    cl_var = np.dot(np.dot(w.T, cov_cl), w)[0, 0]
    return cl_var


def get_hrp(cov, corr):
    """Construct a hierarchical portfolio
    
    Params
    ------
    cov: pd.DataFrame
    corr: pd.DataFrame
    
    Returns
    -------
    pd.Series
    """
    dist = get_corr_dist(corr)
    link = sch.linkage(dist, 'single')
    sort_idx = get_quasi_diag(link)
    # Recover label
    sort_idx = corr.index[sort_idx].tolist()
    hrp = get_rec_bipart(cov, sort_idx)
    return hrp.sort_index()