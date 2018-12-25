import numpy as np
import pandas as pd


def get_evec(dot, var_th):
    """Calculate eigen values and vectors
    
    Params
    ------
    dot: pd.DataFrame
        Z score product dataframe
    var_th: float
        Threshold for the explanation of variance
    
    Returns
    -------
    e_val: pd.Series, eigen values
    e_vec: pd.DataFrame, eigen vectors
    """
    # Compute and sort eigen vectors and values for dot product matrix
    e_val, e_vec = np.linalg.eigh(dot)
    idx = e_val.argsort()[::-1]
    e_val, e_vec = e_val[idx], e_vec[:, idx]
    # Labeling features
    e_val = pd.Series(e_val, index=['PC_' + str(i + 1) for i in range(e_val.shape[0])])
    e_vec = pd.DataFrame(e_vec, index=dot.index, columns=e_val.index)
    e_vec = e_vec.loc[:, e_val.index]
    # Reduce dimension from threshold
    cum_var = e_val.cumsum() / e_val.sum()
    dim = cum_var.searchsorted(var_th)[0]
    e_val = e_val.iloc[:dim + 1]
    e_vec = e_vec.iloc[:, :dim + 1]
    return e_val, e_vec


def ortho_feats(dfX, var_th=.95):
    """Compute orthgonal features with threshold
    
    Params
    ------
    dfX: pd.DataFrame
        Feataures dataframe
    var_th: float
        Threshold for the explanation of variance
        
    Returns
    -------
    pd.DataFrame: orthogonal feature
    """
    Z = (dfX.values - dfX.mean().values) / dfX.std().values
    dot = pd.DataFrame(np.dot(Z.T, Z), index=dfX.columns, columns=dfX.columns)
    e_val, e_vec = get_evec(dot, var_th)
    dfP = pd.DataFrame(np.dot(Z, e_vec), index=dfX.index,
                       columns=['PC_' + str(i + 1) for i in range(e_vec.shape[1])])
    return dfP