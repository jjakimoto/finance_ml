import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.metrics import mutual_info_score

def _fix_corr(corr):
    corr[corr > 1] = 1
    corr[corr < -1] = -1
    return corr.fillna(0)

def corr_metric(corr, use_abs=False):
    corr = _fix_corr(corr)
    if use_abs:
        return np.sqrt(1 - np.abs(corr))
    else:
        return np.sqrt(0.5 * (1 - corr))

def corr_metric_xy(x, y, use_abs=False):
    corr = np.corrcoef(x, y)[0, 1]
    return corr_metric(corr, use_abs)

def _get_zeta(N):
    return (8 + 324 * N + 12 * (36 * N + 729 * N ** 2) ** 0.5) ** (1./3)

def _num_bins(n_obs, corr=None):
    if corr is None or 1. - corr ** 2 < 1e-8:
        zeta = _get_zeta(n_obs)
        b = round(zeta / 6. + 2. / (3 * zeta) + 1. / 3)
    else:
        b = round(2 ** -0.5 * (1 + (1 + 24 * n_obs / (1. - corr ** 2)) ** 0.5) ** 0.5)
    return max(int(b), 2)


def entropy(x, bx=None, is_cont=False):
    if bx is None:
        bx = _num_bins(x.shape[0])
    hx = ss.entropy(np.histogram(x, bx)[0])
    if is_cont:
        delta = (x.max() - x.min()) / bx
        hx += np.log(delta)
    return hx

def joint_entropy(x, y, bxy=None, is_cont=False):
    if bxy is None:
        bxy = _num_bins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])
    cxy = np.histogram2d(x, y, bxy)[0]
    hx = ss.entropy(np.histogram(x, bxy)[0])
    hy = ss.entropy(np.histogram(y, bxy)[0])
    ixy = mutual_info_score(None, None, contingency=cxy)
    hxy = hx + hy - ixy
    if is_cont:
        deltax = (x.max() - x.min()) / bxy
        deltay = (y.max() - y.min()) / bxy
        hxy += np.log(deltax) + np.log(deltay)
    return hxy

def cond_entropy(x, y, bxy=None, is_cont=False):
    if bxy is None:
        bxy = _num_bins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])
    cxy = np.histogram2d(x, y, bxy)[0]
    hx = ss.entropy(np.histogram(x, bxy)[0])
    hy = ss.entropy(np.histogram(y, bxy)[0])
    ixy = mutual_info_score(None, None, contingency=cxy)
    hxy = hx + hy - ixy
    if is_cont:
        deltax = (x.max() - x.min()) / bxy
        deltay = (y.max() - y.min()) / bxy
        hxy += np.log(deltax) + np.log(deltay)
        hy += np.log(deltay)
    return hxy - hy

def variation_info(x, y, normalize=False):
    bxy = _num_bins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])
    cxy = np.histogram2d(x, y, bxy)[0]
    hx = ss.entropy(np.histogram(x, bxy)[0])
    hy = ss.entropy(np.histogram(y, bxy)[0])
    ixy = mutual_info_score(None, None, contingency=cxy)
    varxy = hx + hy - 2 * ixy
    if normalize:
        hxy = hx + hy - ixy
        varxy /= hxy
    return varxy

def mutual_info(x, y, normalize=False):
    bxy = _num_bins(x.shape[0], corr=np.corrcoef(x, y)[0, 1])
    cxy = np.histogram2d(x, y, bxy)[0]
    ixy = mutual_info_score(None, None, contingency=cxy)
    if normalize:
        hx = ss.entropy(np.histogram(x, bxy)[0])
        hy = ss.entropy(np.histogram(y, bxy)[0])
        ixy /= min(hx, hy)
    return ixy