import numpy as np
import pandas as pd


def get_time_decay(tw, last_w=1., truncate=0, is_exp=False):
    cum_w = tw.sort_index().cumsum()
    init_w = 1.
    if is_exp:
        init_w = np.log(init_w)
    if last_w >= 0:
        if is_exp:
            last_w = np.log(last_w)
        slope = (init_w - last_w) / cum_w.iloc[-1]
    else:
        slope = init_w / ((last_w + 1) * cum_w.iloc[-1])
    const = init_w - slope * cum_w.iloc[-1]
    weights = const + slope * cum_w
    if is_exp:
        weights =np.exp(weights)
    weights[weights < truncate] = 0
    return weights


def get_sample_tw(t1, num_co_events, molecule):
    wght = pd.Series(index=molecule)
    for t_in, t_out in t1.loc[wght.index].iteritems():
        wght.loc[t_in] = (1. / num_co_events.loc[t_in: t_out]).mean()
    return wght