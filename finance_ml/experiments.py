import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from sklearn.covariance import LedoitWolf
from sklearn.utils import check_random_state
from sklearn.datasets import make_classification

from .denoising import corr2cov, cov2corr

# Matrix
################################################
def form_block_matrix(n_blocks, bsize, bcorr):
    block = np.ones((bsize, bsize)) * bcorr
    for i in range(bsize):
        block[i, i] = 1
    corr = block_diag(*[block] * n_blocks)
    return corr

def form_true_matrix(n_blocks, bsize, bcorr, is_shuffle=True):
    corr0 = form_block_matrix(n_blocks, bsize, bcorr)
    corr0 = pd.DataFrame(corr0)
    cols = corr0.columns.tolist()
    if is_shuffle:
        np.random.shuffle(cols)
    corr0 = corr0[cols].loc[cols].copy(deep=True)
    std0 = np.random.uniform(0.05, 0.2, corr0.shape[0])
    cov0 = corr2cov(corr0, std0)
    mu0 = np.random.normal(std0, std0, cov0.shape[0]).reshape(-1, 1)
    return mu0, cov0

def simulate_mu_cov(mu, cov, n_obs, shrink=False):
    x = np.random.multivariate_normal(mu.flatten(), cov, size=n_obs)
    mu1 = x.mean(axis=0).reshape(-1, 1)
    if shrink:
        cov1 = LedoitWolf().fit(x).covariance_
    else:
        cov1 = np.cov(x, rowvar=0)
    return mu1, cov1

def get_random_cov(n_cols, n_facts):
    w = np.random.normal(size=(n_cols, n_facts))
    cov = np.dot(w, w.T)
    cov += np.diag(np.random.uniform(size=n_cols))
    return cov

def get_cov_sub(n_obs, n_cols, sigma, random_state=None):
    rng = check_random_state(random_state)
    if n_cols == 1:
        return np.ones((1, 1))
    ar0 = rng.normal(size=(n_obs, 1))
    ar0 = np.repeat(ar0, n_cols, axis=1)
    ar0 += rng.normal(scale=sigma, size=ar0.shape)
    ar0 = np.cov(ar0, rowvar=False)
    return ar0

def get_random_block_cov(n_cols, n_blocks, min_block_size=2, sigma=1., random_state=None):
    rng = check_random_state(random_state)
    # Generate Size of each block
    parts = rng.choice(range(1, n_cols - (min_block_size - 1) * n_blocks), n_blocks-1, replace=False)
    parts.sort()
    parts = np.append(parts, n_cols - (min_block_size - 1) * n_blocks)
    parts = np.append(parts[0], np.diff(parts)) - 1 + min_block_size
    # Combine blocks as diagonal matrix
    cov = None
    for n_cols_ in parts:
        cov_ = get_cov_sub(int(max(n_cols_ * (n_cols_ + 1) / 2., 100)), n_cols_, sigma, random_state=rng)
        if cov is None:
            cov = cov_.copy()
        else:
            cov = block_diag(cov, cov_)
    return cov

def get_random_block_corr(n_cols, n_blocks, random_state=None, min_block_size=2, sigma=1., is_shuffle=False):
    rng = check_random_state(random_state)
    cov0 = get_random_block_cov(n_cols, n_blocks, min_block_size=min_block_size, sigma=sigma * 0.5, random_state=rng)
    # Add noise
    cov1 = get_random_block_cov(n_cols, 1, min_block_size=min_block_size, sigma=sigma, random_state=rng)
    cov0 += cov1
    # Generate Correlation
    corr0 = cov2corr(cov0)
    corr0 = pd.DataFrame(corr0)
    if is_shuffle:
        orig_cols = corr0.columns.tolist()
        cols = corr0.columns.tolist()
        np.random.shuffle(cols)
        corr0 = pd.DataFrame(corr0[cols].loc[cols].values, index=orig_cols, columns=orig_cols)
    return corr0

def get_classification_data(n_features=100, n_informative=25, n_reduntant=25, n_samples=10000,
                            info_sigma=0., red_sigma=0., random_state=0):
    np.random.seed(random_state)
    X, y = make_classification(n_samples=n_samples, n_features=n_features - n_reduntant,
                               n_informative=n_informative, n_redundant=0, shuffle=False)
    cols = [f"I_{i}" for i in range(n_informative)]
    cols += [f"N_{i}" for i in range(n_features - n_reduntant - n_informative)]
    X = pd.DataFrame(X, columns=cols)
    y = pd.Series(y)
    for i in range(n_informative):
        col = f"I_{i}"
        X[col] = X[col] + np.random.normal(size=X.shape[0]) * info_sigma
    rdt_choices = np.random.choice(range(n_informative), size=n_reduntant)
    for i, choice in enumerate(rdt_choices):
        X[f"R_{i}"] = X[f"I_{choice}"] + np.random.normal(size=X.shape[0]) * red_sigma
    return X, y
