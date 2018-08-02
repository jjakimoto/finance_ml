from sklearn.metrics import log_loss, accuracy_score
import numpy as np

from .kfold import PurgedKFold


def cv_score(clf, X, y, sample_weight=None, scoring='neg_log_loss',
             t1=None, n_splits=3, cv_gen=None, pct_embargo=0., purging=False):
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('Wrong scoring method')
    if cv_gen is None:
        cv_gen = PurgedKFold(n_splits=n_splits, t1=t1,
                             pct_embargo=pct_embargo,
                             purging=purging)
    scores = []
    for train, test in cv_gen.split(X=X):
        train_params = dict()
        test_params = dict()
        # Sample weight is an optional parameter
        if sample_weight is not None:
            train_params['sample_weight'] = sample_weight.iloc[train].values
            test_params['sample_weight'] = sample_weight.iloc[test].values
        clf_ = clf.fit(X=X.iloc[train, :], y=y.iloc[train], **train_params)
        # Scoring
        if scoring == 'neg_log_loss':
            prob = clf_.predict_proba(X.iloc[test, :])
            score_ = -log_loss(y.iloc[test], prob, labels=clf.classes_, **test_params)
        else:
            pred = clf_.predict(X.iloc[test, :])
            score_ = accuracy_score(y.iloc[test], pred, **test_params)
        scores.append(score_)
    return np.array(scores)
