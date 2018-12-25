def get_returns(close, timestamps=None, num_days=1):
    """Calculate num_days returns

    Parameters
    ----------
    close: pd.Series
        Close price series
    timestamps: pd.DatetimeIndex, optional
        sampled points to analyze
    num_days: int
        How many days to look forward

    Returns
    -------
    pd.Series: Future price normalized with current price
    """
    if timestamps is None:
        timestamps = close.index
    rets = close.pct_change(num_days).shift(-num_days)
    return rets.loc[timestamps]