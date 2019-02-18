def get_corr_dist(corr):
    """Calculate correlation distance
    
    Params
    ------
    corr: pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
    """
    dist = ((1 - corr) / 2)**.5
    return dist