import numpy as np
import pandas as pd


def get_e_vec(dot, var_thres):
    e_val, e_vec = np.linalg.eigh(dot)
    # Descending order
    idx = e_val.argsort()[::-1]
    e_val = e_val[idx]
    e_vec = e_vec[:, idx]
    # Use only positive ones
    e_val = pd.Series(e_val, index=['PC_' + str(i + 1) for i in range(e_val.shape[0])])
    e_vec = pd.DataFrame(e_vec, index=dot.index, columns=e_val.index)
    e_vec = e_vec.loc[:, e_val > 0]
    e_val = e_val.loc[e_val > 0]
    # Reduce dimension with threashold
    cum_var = e_val.cumsum() / e_val.sum()
    dim = cum_var.values.searchsorted(var_thres)
    e_val = e_val.iloc[:dim+1]
    e_vec = e_vec.iloc[:, :dim+1]
    return e_val, e_vec


def orth_feats(dfX, var_thres=.95):
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)
    e_val, e_vec = get_e_vec(dot, var_thres)
    dfP = pd.DataFrame(np.dot(dfZ, e_vec), index=dfZ.index, columns=e_vec.columns)
    return dfP
