import pandas as pd


def get_train_times(train_times, test_times):
    """Sample train points without overlapping with test period
    
    Params
    ------
    train_times: pd.Series
        Trainig points with index for initial and values for end time
    test_times: pd.Series
        Testing points with index for initial and values for end time
        
    Returns
    -------
    pd.Series
    """
    trn = train_times.copy(deep=True)
    for init, end in test_times.iteritems():
        df0 = trn[(init <= trn.index) & (trn.index <= end)].index
        df1 = trn[(init <= trn) & (trn <= end)].index
        df2 = trn[(trn.index <= init) & (end <= trn)].index
        trn = trn.drop(df0 | df1 | df2)
    return trn


def get_embargo_times(times, pct_embargo):
    """Get embargo time index for each timestamp
    
    times:
        times: Timestamps
            Entire timestamps which you want to apply embargo
        pct_embargo: float ranged at [0, 1]
            The ratio to embargo with respect to the size of timestamps
            
    Returns:
        pd.Series: For each valud corresponds to a point which you should take
        out before from the other forward dataset
    """
    step = int(times.shape[0] * pct_embargo)
    if step == 0:
        embg = pd.Series(times, index=times)
    else:
        embg = pd.Series(times[step:], index=times[:-step])
        embg = embg.append(pd.Series(times[-1], index=times[-step:]))
    return embg
