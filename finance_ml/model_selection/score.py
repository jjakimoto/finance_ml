import numpy as np

from .kfold import PurgedKFold, CPKFold
from .utils import evaluate


def cv_score(clf,
             X,
             y,
             sample_weight=None,
             scoring='neg_log_loss',
             n_splits=3,
             t1=None,
             cv_gen=None,
             pct_embargo=0.,
             purging=True,
             return_combs=False,
             ret=None,
             num_threads=1,
             **kwargs):
    """Cross Validation with default purging and embargo
    
    Params
    ------
    X: pd.DataFrame
    y: pd.Series, optional
    sample_weight: pd.Series, optional
        If specified, apply this to bot testing and training
    scoring: str, default 'neg_log_loss'
        The name of scoring methods. 'precision', 'recall', 'f1', 'precision_recall',
        'roc', 'accuracy' or 'neg_log_loss'
    n_splits: int
        The number of splits for cross validation
    t1: pd.Series
        Index and value correspond to the begining and end of information
    cv_gen: KFold instance
        If not specified, use PurgedKfold. If cv_gen == 'cp', use CPKFold
    pct_embargo: float, default 0
        The percentage of applying embargo
    purging: bool, default True
        If true, apply purging method
    return_combs: bool, default False
        If True and use CPKFold, return combinatorics location
    num_threads: int, default 1
        The number of threads for purging
    kwargs: Parameters for scoring function
        
    Returns
    -------
    array: scores of cross validation
    """
    if cv_gen is None:
        cv_gen = PurgedKFold(
            n_splits=n_splits,
            t1=t1,
            pct_embargo=pct_embargo,
            purging=purging,
            num_threads=num_threads)
    elif cv_gen == 'cp':
        cv_gen = CPKFold(
            n_splits=n_splits,
            t1=t1,
            pct_embargo=pct_embargo,
            purging=purging,
            num_threads=num_threads)
    scores = []
    for train, test in cv_gen.split(X=X):
        train_params = dict()
        test_params = dict()
        # Sample weight is an optional parameter
        if sample_weight is not None:
            train_params['sample_weight'] = sample_weight.iloc[train].values
            test_params['sample_weight'] = sample_weight.iloc[test].values
        test_params.update(kwargs)
        clf_fit = clf.fit(
            X=X.iloc[train, :].values, y=y.iloc[train].values, **train_params)
        if hasattr(clf_fit, 'classes_'):
            test_params['labels'] = clf_fit.classes_
        if ret is not None:
            test_params['ret'] = ret.iloc[test]
        # Scoring
        score_ = evaluate(clf_fit, X.iloc[test, :].values, y.iloc[test].values,
                          scoring, **test_params)
        scores.append(score_)
    if scoring not in ['roc', 'precision_recall']:
        scores = np.array(scores)
    if return_combs:
        return scores, cv_gen.get_test_combs()
    else:
        return scores