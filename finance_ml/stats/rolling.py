import pandas as pd


def pandas_rolling(series, window, freq=1, method='mean'):
    series_list = []
    for i in range(freq):
        _series = series.iloc[i::freq].rolling(window).agg(method)
        series_list.append(_series)
    return pd.concat(series_list, axis=0).sort_index()