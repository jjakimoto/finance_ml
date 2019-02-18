import numbers
from scipy.stats import norm


def drop_labels(events, min_pct=0.05):
    while True:
        df = events['bin'].value_counts(normalize=True)
        if df.min() > min_pct or df.shape[0] < 3:
            break
        print('dropped label', df.argmin(), df.min())
        events = events[events['bin'] != df.argmin()]
    return events


def get_partial_index(df, start=None, end=None):
    """Get partial time index according to start and end

    Parameters
    ----------
    df: pd.DatFrame or pd.Series
    start: str, optional, e.g., '2000-01-01'
    end: str, optional, e.g., '2017-08-31'

    Returns
    -------
    pd.DatetimeIndex
    """
    if start is not None:
        df = df.loc[df.index >= start]
    if end is not None:
        df = df.loc[df.index <= end]
    return df.index


def get_gaussian_betsize(prob, num_classes=2):
    """Translate probability to bettingsize

    Params
    ------
    prob: array-like
    num_classes: int, default 2

    Returns
    -------
    array-like
    """
    if isinstance(prob, numbers.Number):
        if prob != 0 and prob != 1:
            signal = (prob - 1. / num_classes) / (prob * (1 - prob))
        else:
            signal = 2 * prob - 1
    else:
        signal = prob.copy()
        signal[prob == 1] = 1
        signal[prob == 0] = -1
        cond = (prob < 1) & (prob > 0)
        signal[cond] = (prob[cond] - 1. / num_classes) / (prob[cond] *
                                                          (1 - prob[cond]))
    return 2 * norm.cdf(signal) - 1