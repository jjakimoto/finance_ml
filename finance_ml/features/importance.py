import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from finance_ml.multiprocessing import mp_pandas_obj
from finance_ml.model_selection import PurgedKFold, cv_score


def feat_imp_MDI(forest, feat_names):
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


def feat_imp_MDA(clf, X, y, n_splits, sample_weight, t1, pct_embargo,
                 scoring='neg_log_loss'):
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method')
    cv_gen = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)
    index = np.arange(n_splits)
    scores = pd.Series(index=index)
    scores_perm = pd.DataFrame(index=index, columns=X.columns)
    for idx, (train, test) in zip(index, cv_gen.split(X=X)):
        X_train = X.iloc[train]
        y_train = y.iloc[train]
        w_train = sample_weight.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]
        w_test = sample_weight.iloc[test]
        clf_fit = clf.fit(X_train, y_train, sample_weight=w_train.values)
        if scoring == 'neg_log_loss':
            prob = clf_fit.predict_proba(X_test)
            scores.loc[idx] = -log_loss(y_test, prob,
                                        sample_weight=w_test.values,
                                        labels=clf_fit.classes_)
        else:
            pred = clf_fit.predict(X_test)
            scores.loc[idx] = accuracy_score(y_test, pred,
                                             sample_weight=w_test.values)

        for col in X.columns:
            X_test_ = X_test.copy(deep=True)
            # Randomize certain feature to make it not effective
            np.random.shuffle(X_test_[col].values)
            if scoring == 'neg_log_loss':
                prob = clf_fit.predict_proba(X_test_)
                scores_perm.loc[idx, col] = -log_loss(y_test, prob,
                                                      sample_weight=w_test.value,
                                                      labels=clf_fit.classes_)
            else:
                pred = clf_fit.predict(X_test_)
                scores_perm.loc[idx, col] = accuracy_score(y_test, pred,
                                                           sample_weight=w_test.values)
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


def aux_feat_imp_SFI(feat_names, clf, X, cont, scoring, cv_gen):
    imp = pd.DataFrame(columns=['mean', 'std'])
    for feat_name in feat_names:
        scores = cv_score(clf, X=X[[feat_name]], y=cont['bin'],
                          sample_weight=cont['w'],
                          scoring=scoring,
                          cv_gen=cv_gen)
        imp.loc[feat_name, 'mean'] = scores.mean()
        imp.loc[feat_name, 'std'] = scores.std() * np.sqrt(scores.shape[0])
    return imp


def feat_importance(X, cont, clf=None, n_estimators=1000, n_splits=10, max_samples=1.,
                    num_threads=24, pct_embargo=0., scoring='accuracy',
                    method='SFI', min_w_leaf=0., **kwargs):
    n_jobs = (-1 if num_threads > 1 else 1)
    # Build classifiers
    if clf is None:
        base_clf = DecisionTreeClassifier(criterion='entropy', max_features=1,
                                          class_weight='balanced',
                                          min_weight_fraction_leaf=min_w_leaf)
        clf = BaggingClassifier(base_estimator=base_clf, n_estimators=n_estimators,
                                max_features=1., max_samples=max_samples,
                                oob_score=True, n_jobs=n_jobs)
    fit_clf = clf.fit(X, cont['bin'], sample_weight=cont['w'].values)
    if hasattr(fit_clf, 'oob_score_'):
        oob = fit_clf.oob_score_
    else:
        oob = None
    if method == 'MDI':
        imp = feat_imp_MDI(fit_clf, feat_names=X.columns)
        oos = cv_score(clf, X=X, y=cont['bin'], n_splits=n_splits,
                       sample_weight=cont['w'], t1=cont['t1'],
                       pct_embargo=pct_embargo, scoring=scoring).mean()
    elif method == 'MDA':
        imp, oos = feat_imp_MDA(clf, X=X, y=cont['bin'], n_splits=n_splits,
                                sample_weight=cont['w'], t1=cont['t1'],
                                pct_embargo=pct_embargo, scoring=scoring)
    elif method == 'SFI':
        cv_gen = PurgedKFold(n_splits=n_splits, t1=cont['t1'], pct_embargo=pct_embargo)
        oos = cv_score(clf, X=X, y=cont['bin'], sample_weight=cont['w'],
                       scoring=scoring, cv_gen=cv_gen)
        clf.n_jobs = 1
        imp = mp_pandas_obj(aux_feat_imp_SFI, ('feat_names', X.columns),
                            num_threads, clf=clf, X=X, cont=cont,
                            scoring=scoring, cv_gen=cv_gen)
    return imp, oob, oos