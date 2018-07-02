import numpy as np
import pandas as pd


def get_weights(d, size):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def frac_diff(series, d, thres=.1):
    # w.shape = (series.shape[0], 1)
    w = get_weights(d, series.shape[0])
    w_sum = np.cumsum(abs(w))
    w_sum /= w_sum[-1]
    # Usable only after going over the threshold
    skip = w_sum[w_sum > thres].shape[0]
    df = {}
    for name in series.columns:
        series_f = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series()
        for iloc in range(skip, series_f.shape[0]):
            loc = series_f.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
                continue
            df_[loc] = np.dot(w[-(iloc + 1):, :].T, series_f.loc[:loc])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def get_weights_FFD(d, thres, max_size=10000):
    w = [1.]
    for k in range(1, max_size):
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) <= thres:
            break
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def frac_diff_FFD(series, d, thres=1e-5):
    w = get_weights_FFD(d, thres)
    width = len(w) - 1
    df = {}
    for name in series.columns:
        series_f = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series()
        for iloc1 in range(width, series_f.shape[0]):
            loc0 = series_f.index[iloc1 - width]
            loc1 = series_f.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue
            df_[loc1] = np.dot(w.T, series_f.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df