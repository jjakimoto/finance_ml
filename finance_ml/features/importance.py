import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from finance_ml.multiprocessing import mp_pandas_obj
from finance_ml.model_selection import PurgedKFold, cv_score, evaluate


def feat_imp_MDI(forest, feat_names):
    """Compute Mean Decrease Impurity
    
    Params
    ------
    forest: Forest Classifier instance
    feat_names: list(str)
        List of names of features

    Returns
    -------
    imp: pd.DataFrame
        Importance means and standard deviations
    """
    imp_dict = {i: tree.feature_importances_ for i, tree in
                enumerate(forest.estimators_)}
    imp_df = pd.DataFrame.from_dict(imp_dict, orient='index')
    imp_df.columns = feat_names
    # 0 simply means not used for splitting
    imp_df = imp_df.replace(0, np.nan)
    imp = pd.concat({'mean': imp_df.mean(),
                     'std': imp_df.std() * np.sqrt(imp_df.shape[0])},
                    axis=1)
    imp /= imp['mean'].sum()
    return imp


def feat_imp_MDA(clf, X, y, sample_weight=None, scoring='neg_log_loss', n_splits=3, t1=None,
                 cv_gen=None, pct_embargo=0, purging=True, num_threads=1):
    """Calculate Mean Decrease Accuracy
    
    Params
    ------
    clf: Classifier instance
    X: pd.DataFrame, Input feature
    y: pd.Series, Label        
    sample_weight: pd.Series, optional
        If specified, apply this to bot testing and training
    scoring: str, default 'neg_log_loss'
        The name of scoring methods. 'f1', 'accuracy' or 'neg_log_loss'
    n_splits: int, default 3
        The number of splits for cross validation
    t1: pd.Series
        Index and value correspond to the begining and end of information
    cv_gen: KFold instance
        If not specified, use PurgedKfold
    pct_embargo: float, default 0
        The percentage of applying embargo
    purging: bool, default True
        If true, apply purging method
    num_threads: int, default 1
        The number of threads for purging
    
    Returns
    -------
    imp: pd.DataFrame, feature importance of means and standard deviations
    scores: float, scores of cross validation
    """
    
    if cv_gen is None:
        cv_gen = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo,
                             purging=purging, num_threads=num_threads)
    index = np.arange(n_splits)
    scores = pd.Series(index=index)
    scores_perm = pd.DataFrame(index=index, columns=X.columns)
    for idx, (train, test) in zip(index, cv_gen.split(X=X)):
        X_train = X.iloc[train]
        y_train = y.iloc[train]
        if sample_weight is not None:
            w_train = sample_weight.iloc[train].values
        else:
            w_train = None
        X_test = X.iloc[test]
        y_test = y.iloc[test]
        if sample_weight is not None:
            w_test = sample_weight.iloc[test].values
        else:
            w_test = None
        clf_fit = clf.fit(X_train, y_train, sample_weight=w_train)
        scores.loc[idx] = evaluate(clf_fit, X_test, y_test, scoring,
                                   sample_weight=w_test, labels=clf_fit.classes_)

        for col in X.columns:
            X_test_ = X_test.copy(deep=True)
            # Randomize certain feature to make it not effective
            np.random.shuffle(X_test_[col].values)
            scores_perm.loc[idx, col] = evaluate(clf_fit, X_test_, y_test, scoring,
                                                 sample_weight=w_test, labels=clf_fit.classes_)
    # (Original score) - (premutated score)
    imprv = (-scores_perm).add(scores, axis=0)
    # Relative to maximum improvement
    if scoring == 'neg_log_loss':
        max_imprv = -scores_perm
    else:
        max_imprv = 1. - scores_perm
    imp = imprv / max_imprv
    imp = pd.DataFrame(
        {'mean': imp.mean(), 'std': imp.std() * np.sqrt(imp.shape[0])})
    return imp, scores.mean()


def mp_feat_imp_SFI(clf, X, y, feat_names, sample_weight=None, scoring='neg_log_loss',
                    n_splits=3, t1=None, cv_gen=None, pct_embargo=0, purging=True):
    imp = pd.DataFrame(columns=['mean', 'std'])
    for feat_name in feat_names:
        scores = cv_score(clf, X=X[[feat_name]], y=y,
                          sample_weight=sample_weight,
                          scoring=scoring,
                          cv_gen=cv_gen,
                          n_splits=n_splits,
                          t1=t1,
                          pct_embargo=pct_embargo,
                          purging=purging)
        imp.loc[feat_name, 'mean'] = scores.mean()
        imp.loc[feat_name, 'std'] = scores.std() * np.sqrt(scores.shape[0])
    return imp


def feat_imp_SFI(clf, X, y, sample_weight=None, scoring='neg_log_loss',
                 n_splits=3, t1=None, cv_gen=None, pct_embargo=0, purging=True, num_threads=1):
    """Calculate Single Feature Importance
    
    Params
    ------
    clf: Classifier instance
    X: pd.DataFrame
    y: pd.Series, optional
    sample_weight: pd.Series, optional
        If specified, apply this to bot testing and training
    scoring: str, default 'neg_log_loss'
        The name of scoring methods. 'accuracy' or 'neg_log_loss'
    
    n_splits: int, default 3
        The number of splits for cross validation
    t1: pd.Series
        Index and value correspond to the begining and end of information
    cv_gen: KFold instance
        If not specified, use PurgedKfold
    pct_embargo: float, default 0
        The percentage of applying embargo
    purging: bool, default True
        If true, apply purging method
    num_threads: int, default 1
        The number of threads for multiprocessing multi features
        
    Returns
    -------
    imp: pd.DataFrame, feature importance of means and standard deviations
    """
    imp = mp_pandas_obj(mp_feat_imp_SFI, ('feat_names', X.columns),
                        num_threads, clf=clf, X=X, y=y, sample_weight=sample_weight,
                        scoring=scoring, n_splits=n_splits, t1=t1, cv_gen=cv_gen,
                        pct_embargo=pct_embargo, purging=purging)
    return imp


