import numpy as np
import pandas as pd

def get_bsadf(series, tau, constant, lags):
    y, x = get_yx(series, constant=constant, lags=lags)
    if not isinstance(lags, int):
        lags = np.max(lags)
    start_points = range(0, y.shape[0] - tau + 1)
    basdf = None
    all_adf = []
    for start in start_points:
        y_ = y[start:]
        x_ = x[start:]
        b_mean, b_var = get_betas(y_, x_)
        b_mean = b_mean[0,]
        b_std = b_var[0, 0] ** 0.5
        all_adf.append(b_mean / b_std)
    all_adf = np.array(all_adf)
    bsadf = np.max(all_adf[np.isfinite(all_adf)])
    out = {'Time': series.index[-1], 'bsadf': bsadf}
    return out


def get_yx(series, constant, lags):
    diff = series.diff().dropna()
    lag_feat = get_lag_features(diff, lags).dropna()
    # Add non diff feature
    lag_feat[series.name] = series.shift(1)
    index = lag_feat.dropna().index & diff.dropna().index
    x = lag_feat.loc[index].values
    y = diff.loc[index].values
    # Set constant value
    if constant != 'nc':
        const = np.ones((x.shape[0], 1))
        x = np.hstack((x, const))
        if constant[:2] == 'ct':
            trend = np.arange(x.shape[0]).reshape(-1, 1)
            x = np.hstack((x, trend))
        if constant == 'ctt':
            x = np.hstack((x, trend ** 2))
    return y, x


def get_lag_features(series, lags):
    lag_feat = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(1, lags + 1)
    else:
        lags = [int(lag) for lag in lags]
    for lag in lags:
        lag_feat[f'{series.name}_{lag}'] = series.shift(lag).copy(deep=True)
    return lag_feat

def get_betas(y, x, lam=0):
    xy = np.dot(x.T, y)
    xx = np.dot(x.T, x)
    xxinv = np.linalg.inv(xx + lam)
    beta_mean = np.dot(xxinv, xy)
    err = y - np.dot(x, beta_mean)
    beta_var = np.dot(err.T, err) / (x.shape[0] - x.shape[1]) * xxinv
    return beta_mean, beta_var