def calc_prob_sr(pt, sl, freq, tgt_sr, rf=0.):
    """Calculate required probability wrt target SR

    Paramters
    ---------
    pt: float
        Profit Take
    sl: float
        Stop Loss
    freq: float
        Frequency of trading
    tgt_sr: float
        Target Sharpe Ratio
    rf: float, (default 0)
        Risk Free Rate

    Returns
    -------
    float: Required probability
    """
    diff = pt - sl
    a = (freq + tgt_sr ** 2) * diff ** 2
    b = diff * (2 * freq * (sl - rf) - tgt_sr ** 2 * diff)
    c = freq * (sl - rf) ** 2
    p = (-b + (b ** 2 - 4 * a * c) ** .5) / (2. * a)
    return p


def prob_failure(ret, freq, tgt_sr):
    """
    Calculate the probability to fail in achieving
    target Sharpe Ratio

    Parameters
    ----------
    ret: array-like
        Returns of trading
    freq: float
        Frequency of trading
    tgt_sr: float
        Aiming Sharpe Ratio

    Returns
    -------
    risk: float
    """
    r_pos = ret[ret > 0].mean()
    r_neg = ret[ret <= 0].mean()
    p = ret[ret > 0].shape[0] / float(ret.shape[0])
    th_p = calc_prob_sr(r_pos, r_neg, freq, tgt_sr)
    risk = ss.norm.cdf(th_p, p, p * (1 - p))
    return risk