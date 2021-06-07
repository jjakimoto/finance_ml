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

    Args:
        df (pd.DatFrame or pd.Series)

        start (datetime.datetime, optional): e.g., datetime(2018, 1, 1)

        end (datetime.datetime, optional): e.g., dateteim(2018, 3, 1)

    Returns:
        pd.DatetimeIndex
    """
    if start is not None:
        df = df.loc[df.index >= start]
    if end is not None:
        df = df.loc[df.index <= end]
    return df.index


def get_gaussian_betsize(probs, num_classes=2):
    """Translate probability to bettingsize

    Args:
        probs (array-like)
        
        num_classes (int, optional): Defaults to 2

    Returns:
        array-like: Signals after gaussian transform
    """
    if isinstance(probs, numbers.Number):
        if probs != 0 and probs != 1:
            signal = (probs - 1. / num_classes) / (probs * (1 - probs))
        else:
            signal = 2 * probs - 1
    else:
        signal = probs.copy()
        signal[probs == 1] = 1
        signal[probs == 0] = -1
        cond = (probs < 1) & (probs > 0)
        signal[cond] = (probs[cond] - 1. / num_classes) / (probs[cond] * (1 - probs[cond]))
    return 2 * norm.cdf(signal) - 1