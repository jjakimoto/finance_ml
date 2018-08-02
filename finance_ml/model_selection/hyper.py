from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier

from .kfold import PurgedKFold
from .pipeline import Pipeline


def clf_hyper_fit(feat, label, t1, pipe_clf, search_params, scoring=None,
                  n_splits=3, bagging=[0, None, 1.],
                  rnd_search_iter=0, n_jobs=-1, pct_embargo=0., **fit_params):
    # Set default value for scoring
    if scoring is None:
        if set(label.values) == {0, 1}:
            scoring = 'f1'
        else:
            scoring = 'neg_log_loss'
    # HP search on training data
    inner_cv = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)
    if rnd_search_iter == 0:
        search = GridSearchCV(estimator=pipe_clf, param_grid=search_params,
                              scoring=scoring, cv=inner_cv, n_jobs=n_jobs, iid=False)
    else:
        search = RandomizedSearchCV(estimator=pipe_clf, param_distributions=search_params,
                                    scoring=scoring, cv=inner_cv, n_jobs=n_jobs, iid=False)
    best_pipe = search.fit(feat, label, **fit_params).best_estimator_
    # Fit validated model on the entirely of data
    if bagging[0] > 0:
        bag_est = BaggingClassifier(base_estimator=Pipeline(best_pipe.steps),
                                    n_estimators=int(bagging[0]), max_samples=float(bagging[1]),
                                    max_features=float(bagging[2]), n_jobs=n_jobs)
        bag_est = best_pipe.fit(feat, label,
                                sample_weight=fit_params[bag_est.base_estimator.steps[-1][0] + '__sample_weight'])
        best_pipe = Pipeline([('bag', bag_est)])
    return best_pipe