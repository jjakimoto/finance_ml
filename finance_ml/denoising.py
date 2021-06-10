import numpy as np
import pandas as pd
from sklearn.neighbors.kde import KernelDensity
from scipy.optimize import minimize

from .utils import cov2corr, corr2cov


def mp_pdf(var, q, pts):
    # Marcenko-Pastur Distribution
    # q = T/N
    e_min = var * (1 - (1./q) ** 0.5) ** 2
    e_max = var * (1 + (1./q) ** 0.5) ** 2
    e_val = np.linspace(e_min, e_max, pts)
    pdf = q * ((e_max - e_val) * (e_val - e_min)) ** 0.5 / (2 * np.pi * var * e_val)
    return pd.Series(pdf, index=e_val)

def getPCA(matrix):
    e_val, e_vec = np.linalg.eigh(matrix)
    indices = e_val.argsort()[::-1]
    e_val = e_val[indices]
    e_vec = e_vec[:, indices]
    e_val = np.diagflat(e_val)
    return e_val, e_vec

def fitKDE(obs, bwidth=0.25, kernel='gaussian', x=None):
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)
    kde = KernelDensity(kernel=kernel, bandwidth=bwidth).fit(obs)
    if x is None:
        x = np.unique(obs).reshape(-1 , 1)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    log_prob = kde.score_samples(x)
    pdf = pd.Series(np.exp(log_prob), index=x.flatten())
    return pdf

def err_pdf(var, e_val, q, bwidth, pts=1000):
    pdf0 = mp_pdf(var[0], q, pts)
    pdf1 = fitKDE(e_val, bwidth, x=pdf0.index.values)
    sse = np.sum((pdf1 - pdf0) ** 2)
    return sse

def find_max_eigen_val(e_val, q, bwidth, min_var=1e-5, max_var=1-1e-5):
    out = minimize(lambda *x: err_pdf(*x), .5, args=(e_val, q, bwidth), bounds=((min_var, max_var),))
    if out["success"]:
        var = out['x'][0]
    else:
        var = 1
    e_max = var * (1 + (1./q) ** 0.5) ** 2
    return e_max, var


def denoise_corr(e_val, e_vec, n_facts, shrinkage=False, alpha=0):
    if shrinkage:
        e_val_l, e_vec_l = e_val[:n_facts, :n_facts], e_vec[:, :n_facts]
        e_val_r, e_vec_r = e_val[n_facts:, n_facts:], e_vec[:, n_facts:]
        corr_l = np.dot(e_vec_l, e_val_l).dot(e_vec_l.T)
        corr_r = np.dot(e_vec_r, e_val_r).dot(e_vec_r.T)
        corr1 = corr_l + alpha * corr_r + (1 - alpha) * np.diag(np.diag(corr_r))
    else:
        e_val_ = np.diag(e_val).copy()
        e_val_[n_facts:] = e_val_[n_facts:].sum() / float(e_val_.shape[0] - n_facts)
        e_val_ = np.diag(e_val_)
        corr1 = np.dot(e_vec, e_val_).dot(e_vec.T)
        # Renormalize to keep trace 1
        corr1 = cov2corr(corr1)
    return corr1


def detone_corr(e_val, e_vec, n_facts, shrinkage=False, alpha=0):
    if shrinkage:
        e_val_r, e_vec_r = e_val[n_facts:, n_facts:], e_vec[:, n_facts:]
        corr_r = np.dot(e_vec_r, e_val_r).dot(e_vec_r.T)
        corr1 = alpha * corr_r + (1 - alpha) * np.diag(np.diag(corr_r))
        # Renormalize to keep trace 1
        corr1 = cov2corr(corr1)
    else:
        e_val_ = np.diag(e_val).copy()
        e_val_[:n_facts] = 0
        e_val_ = np.diag(e_val_)
        corr1 = np.dot(e_vec, e_val_).dot(e_vec.T)
        # Renormalize to keep trace 1
        corr1 = cov2corr(corr1)
    return corr1


def denoise_cov(cov, q, bwidth):
    corr0 = cov2corr(cov)
    e_val0, e_vec0 = getPCA(corr0)
    e_max0, var0 = find_max_eigen_val(np.diag(e_val0), q, bwidth)
    nfacts0 = e_val0.shape[0] - np.diag(e_val0)[::-1].searchsorted(e_max0)
    corr1 = denoise_corr(e_val0, e_vec0, nfacts0)
    cov1 = corr2cov(corr1, np.diag(cov) ** .5)
    return cov1

def opt_portfolio(cov, mu=None):
    inv = np.linalg.inv(cov)
    ones = np.ones(shape=(inv.shape[0], 1))
    if mu is None:
        mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)
    return w
