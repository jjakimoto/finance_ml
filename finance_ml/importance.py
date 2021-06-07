import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, mean_squared_error


def _neg_log_loss(y, prob, labels=None):
    return -log_loss(y, prob, labels=labels)

def feat_imp_mdi(fit, feat_names):
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = feat_names
    df0 = df0.replace(0, np.nan)
    imp = pd.concat({"mean": df0.mean(), "std": df0.std() * (df0.shape[0] ** -0.5)}, axis=1)
    imp /= imp["mean"].sum()
    return imp

def feat_imp_mda(clf, X, y, metric=None, is_clf=True, n_splits=10):
    cv = KFold(n_splits=n_splits)
    scr0 = pd.Series()
    scr1 = pd.DataFrame()
    if metric is None:
        if is_clf:
            metric = _neg_log_loss
        else:
            metric = mean_squared_error
    for i, (train, test) in enumerate(cv.split(X=X)):
        x0, y0 = X.iloc[train, :], y.iloc[train]
        x1, y1 = X.iloc[test, :], y.iloc[test]
        fit = clf.fit(X=x0, y=y0)
        if is_clf:
            prob = fit.predict_proba(x1)
            scr0.loc[i] = metric(y1, prob, labels=clf.classes_)
        else:
            pred = fit.predict(x1)
            scr0.loc[i] = metric(y1, pred)
        for col in X.columns:
            x1_ = x1.copy(deep=True)
            np.random.shuffle(x1_[col].values)
            if is_clf:
                prob = fit.predict_proba(x1_)
                scr1.loc[i, col] = metric(y1, prob, labels=clf.classes_)
            else:
                pred = fit.predict(x1_)
                scr1.loc[i, col] = metric(y1, pred)
    imp = (-1 * scr1).add(scr0, axis=0)
    imp = imp / (-1 * scr1)
    imp = pd.concat({"mean": imp.mean(), "std": imp.std() * (imp.shape[0] ** -0.5)}, axis=1)
    return imp

def group_mean_std(df0, clstrs):
    out = pd.DataFrame(columns=['mean', 'std'])
    for key, elements in clstrs.items():
        df1 = df0[elements].sum(axis=1)
        out.loc[f"C_{key}", 'mean'] = df1.mean()
        out.loc[f"C_{key}", 'std'] = df1.std() * df1.shape[0]**-.5
    return out

def feat_imp_mdi_clustered(fit, featNames, clstrs):
    df0 = {i:tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan) #because max_features=1
    imp = group_mean_std(df0, clstrs)
    imp /= imp['mean'].sum()
    return imp

def feat_imp_mda_clustered(clf, X, y, clstrs, metric=None, is_clf=True, n_splits=10):
    cv_gen = KFold(n_splits=n_splits)
    scr0 = pd.Series()
    scr1 = pd.DataFrame(columns=clstrs.keys())
    if metric is None:
        if is_clf:
            metric = _neg_log_loss
        else:
            metric = mean_squared_error
    for i, (train, test) in enumerate(cv_gen.split(X=X)):
        x0, y0 = X.iloc[train, :], y.iloc[train]
        x1, y1 = X.iloc[test, :], y.iloc[test]
        fit = clf.fit(X=x0, y=y0)
        if is_clf:
            prob = fit.predict_proba(x1)
            scr0.loc[i] = metric(y1, prob, labels=clf.classes_)
        else:
            pred = fit.predict(x1)
            scr0.loc[i] = metric(y1, pred)
        for col in scr1.columns:
            x1_ = x1.copy(deep=True)
            for k in clstrs[col]:
                np.random.shuffle(x1_[k].values)
            if is_clf:
                prob = fit.predict_proba(x1_)
                scr1.loc[i, col] = metric(y1, prob, labels=clf.classes_)
            else:
                pred = fit.predict(x1_)
                scr1.loc[i, col] = metric(y1, pred)
    imp = (-1 * scr1).add(scr0, axis=0)
    imp = imp / (-1 * scr1)
    imp = pd.concat({'mean': imp.mean(), 'std': imp.std() * imp.shape[0] ** -0.5}, axis=1)
    imp.index = [f"C_{i}" for i in imp.index]
    return imp