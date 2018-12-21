import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold

from .utils import get_train_times


class PurgedKFold(_BaseKFold):
    """Cross Validation with purging and embargo
    
    Params
    ------
    n_splits: int
        The number of splits for cross validation
    t1: pd.Series
        Index and value correspond to the begining and end of information
    pct_embargo: float, default 0
        The percentage of applying embargo
    purging: bool, default True
        If true, apply purging method
    """
    
    def __init__(self, n_splits=3, t1=None, pct_embargo=0., purging=True):
        super(PurgedKFold, self).__init__(n_splits=n_splits, shuffle=False, random_state=None)
        if not isinstance(t1, pd.Series):
            raise ValueError('t1 must be pd.Series')
        self.t1 = t1
        self.pct_embargo = pct_embargo
        self.purging = purging

    def split(self, X, y=None, groups=None):
        """Get train and test times stamps
        
        Params
        ------
        X: pd.DataFrame
        y: pd.Series, optional
        
        Returns
        -------
        train_indices, test_indices: np.array
        """
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError('X and t1 must have the same index')
        indices = np.arange(X.shape[0])
        # Embargo width
        embg_size = int(X.shape[0] * self.pct_embargo)
        # Pandas is close set when using [t0:t1]
        test_ranges = [(i[0], i[-1] + 1) for i in
                       np.array_split(indices, self.n_splits)]
        for st, end in test_ranges:
            test_indices = indices[st:end]
            t0 = self.t1.index[st]
            # Avoid look ahead leakage here
            train_indices = self.t1.index.searchsorted(
                self.t1[self.t1 <= t0].index)
            # Edge point of test set in the most recent side
            max_t1_idx = self.t1.index.searchsorted(
                self.t1[test_indices].max())
            if max_t1_idx < X.shape[0]:
                # Adding indices after test set
                train_indices = np.concatenate(
                    (train_indices, indices[max_t1_idx + embg_size:]))
            # Purging
            if self.purging:
                train_t1 = self.t1.iloc[train_indices]
                test_t1 = self.t1.iloc[test_indices]
                train_t1 = get_train_times(train_t1, test_t1)
                train_indices = self.t1.index.searchsorted(train_t1.index)
            yield train_indices, test_indices