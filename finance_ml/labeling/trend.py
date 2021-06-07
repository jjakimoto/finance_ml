import numpy as np
import pandas as pd

import statsmodels.api as sm

from ..multiprocessing import mp_pandas_obj


def t_val_linreg(close):
    x = np.ones((close.shape[0], 2))
    x[:, 1] = np.arange(close.shape[0])
    ols = sm.OLS(close, x).fit()
    return ols.tvalues[1]

def _get_bins_from_trend(molecule, close, span):
    out = pd.DataFrame(index=molecule, columns=['t1', 't_val','bin'])
    hrzns = list(range(*span))
    for dt0 in molecule:
        iloc0 = close.index.get_loc(dt0)
        if iloc0 + max(hrzns) > close.shape[0]:
            continue
        df0 = pd.Series()
        for hrzn in hrzns:
            dt1 = close.index[iloc0 + hrzn - 1]
            df1 = close.loc[dt0:dt1]
            df0.loc[dt1] = t_val_linreg(df1.values)
        # Get maximum tstats point
        dt1 = df0.replace([-np.inf, np.inf, np.nan], 0).abs().idxmax()
        out.loc[dt0, ['t1', 't_val', 'bin']] = df0.index[-1], df0[dt1], np.sign(df0[dt1])
    out['t1'] = pd.to_datetime(out['t1'])
    out['bin'] = pd.to_numeric(out['bin'], downcast='signed')
    return out.dropna(subset=['bin'])


def get_bins_from_trend(close, span, num_threads=1):
    output = mp_pandas_obj(func=_get_bins_from_trend,
                             pd_obj=('molecule', close.index),
                             num_threads=num_threads,
                             close=close, span=span)
    return output


    