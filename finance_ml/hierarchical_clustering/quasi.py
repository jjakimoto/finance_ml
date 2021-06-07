import numpy as np
import pandas as pd


def get_quasi_diag(link):
    """Calculate quasi diagonalization
    
    Params
    ------
    link: list
        Result from hierachical clustering of scipy
        
    Returns
    -------
    pd.Series: sorted index
    """
    # Make labels integers
    link = link.astype(int)
    sort_idx = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    # Iterate until all elements are assigned
    while sort_idx.max() >= num_items:
        # Gerante index for the first element of cluster
        sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
        # Get clustered value not single elements
        clusters = sort_idx[sort_idx >= num_items]
        idx = clusters.index
        # Add clusters
        cl_idx = clusters.values - num_items
        sort_idx[idx] = link[cl_idx, 0]
        df = pd.Series(link[cl_idx, 1], index=idx + 1)
        sort_idx = sort_idx.append(df)
        # Resort
        sort_idx = sort_idx.sort_index()
        sort_idx.index = range(sort_idx.shape[0])
    return sort_idx.tolist()