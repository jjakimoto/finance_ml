import numbers


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
