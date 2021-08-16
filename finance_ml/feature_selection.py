from copy import deepcopy
import warnings

import numpy as np
import pandas as pd

from .importance import feat_imp_MDI, feat_imp_MDA
from .importance import feat_imp_MDA_clustered, feat_imp_MDI_clustered
from .model_selection import cv_score
from .clustering import cluster_kmeans_top


def _select_features(model, features, labels, q, mode, **kwargs):
    if mode.lower() == 'mdi':
        model.fit(features, labels)
        imp = feat_imp_MDI(model, features.columns)["mean"]
    elif mode.lower() == "mda":
        imp = feat_imp_MDA(model, features, labels, **kwargs)["mean"]
    else:
        raise ValueError(f"Unknown mode: {mode}")    
    score_th = imp.quantile(q)
    return imp[imp > score_th].index

def _select_features_clustered(model, features, labels, q, mode,
                               min_num_clusters=4, max_num_clusters=10,
                               n_init=20, **kwargs):
    warnings.simplefilter('ignore')
    try:
        corr_clstrs, clstrs, silh = cluster_kmeans_top(features.corr(),
                                                   min_num_clusters=min_num_clusters,
                                                   max_num_clusters=max_num_clusters,
                                                   n_init=n_init)
    except Exception as err:
        print(err)
        return features.columns
    if mode.lower() == 'mdi':
        model.fit(features, labels)
        imp = feat_imp_MDI_clustered(model, features.columns, clstrs)["mean"]
    elif mode.lower() == "mda":
        imp = feat_imp_MDA_clustered(model, features, labels, clstrs, **kwargs)["mean"]
    else:
        raise ValueError(f"Unknown mode: {mode}")    
    score_th = imp.quantile(q)
    selected_clstrs = imp[imp > score_th].index
    selected_columns = list()
    for x in selected_clstrs:
        clstr_i = int(x.split("_")[-1])
        selected_columns += clstrs[clstr_i]
    return selected_columns

def select_features(model, features, labels, q=0.1, n_splits=5, scoring="neg_log_loss", mode="mdi",
                    num_round=10, early_stop=True, use_cluster=False, verbose=1, **kwargs):
    scores = cv_score(model, features, labels, n_splits=n_splits, scoring=scoring)
    curr_score = np.mean(scores)
    selected_columns = features.columns
    best_columns = deepcopy(selected_columns)
    trial_count = 0
    while trial_count < num_round and len(selected_columns) > 1:
        trial_count += 1
        select_kwargs = {
            "n_splits": n_splits,
            "scoring": scoring
        }
        if use_cluster:
            select_kwargs.update(kwargs)
            new_columns = _select_features_clustered(model, features[selected_columns], labels, q, mode, **select_kwargs)
        else:
            new_columns = _select_features(model, features[selected_columns], labels, q, mode, **select_kwargs)
        scores = cv_score(model, features[new_columns], labels, n_splits=n_splits, scoring=scoring)
        new_score = np.mean(scores)
        if verbose > 0:
            print(f"Round{trial_count}: best_score={curr_score}, score@{len(new_columns)}={new_score}")
        if new_score > curr_score:
            curr_score = new_score
            best_columns = deepcopy(new_columns)
        elif early_stop:
            break
        selected_columns = new_columns
    return best_columns, curr_score