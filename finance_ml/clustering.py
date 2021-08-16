import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from .distance import corr_metric


_eps = 1e-16

def cluster_kmeans_base(corr0, max_num_clusters=10, min_num_clusters=2, n_init=10, debug=False):
    dist = corr_metric(corr0, False)
    silh = None
    kmeans = None
    q_val = None
    max_num_clusters = min(max_num_clusters, int(np.floor(dist.shape[0]/2)))
    min_num_clusters = max(2, min_num_clusters)
    for init in range(n_init):
        for n_clusters in range(min_num_clusters, max_num_clusters + 1):
            kmeans_ = KMeans(n_clusters=n_clusters, n_jobs=1, n_init=1, random_state=init)
            kmeans_ = kmeans_.fit(dist.values)
            silh_ = silhouette_samples(dist.values, kmeans_.labels_)
            # q_val_ = silh_.mean() / max(silh_.std(), _eps)
            q_val_ = silh_.mean()
            if q_val is None or q_val_ > q_val:
                silh = silh_
                kmeans = kmeans_
                q_val = q_val_
                if debug:
                    print(kmeans)
                    print(q_val, silh)
                    silhouette_avg = silhouette_score(dist.values, kmeans_.labels_)
                    print(f"For n_clusters={n_clusters}, slih_std: {silh_.std()} The average silhouette_score is : {silhouette_avg}")
                    print("********")
    new_idx = np.argsort(kmeans.labels_)
    corr1 = corr0.iloc[new_idx]
    corr1 = corr1.iloc[:, new_idx]
    clstrs = {i:corr0.columns[np.where(kmeans.labels_ == i)[0]].tolist() for i in np.unique(kmeans.labels_)}
    silh = pd.Series(silh, index=dist.index)
    return corr1, clstrs, silh

def make_new_outputs(corr0, clstrs1, clstrs2):
    clstrs_new = dict()
    for i in clstrs1.keys():
        clstrs_new[len(clstrs_new.keys())] = list(clstrs1[i])
    for i in clstrs2.keys():
        clstrs_new[len(clstrs_new.keys())] = list(clstrs2[i])
    new_idx = [j for i in clstrs_new.keys() for j in clstrs_new[i]]
    corr_new = corr0.loc[new_idx, new_idx]
    dist = corr_metric(corr0, False)
    kmeans_labels = np.zeros(len(dist.columns))
    for i in clstrs_new.keys():
        idxs = [dist.index.get_loc(k) for k in clstrs_new[i]]
        kmeans_labels[idxs] = i
    silh_new = pd.Series(silhouette_samples(dist.values, kmeans_labels), index=dist.index)
    return corr_new, clstrs_new, silh_new

def cluster_kmeans_top(corr0, max_num_clusters=None, min_num_clusters=4, n_init=10, debug=False):
    if max_num_clusters is None:
        max_num_clusters = corr0.shape[1] - 1
    max_num_clusters = min(max_num_clusters, corr0.shape[1] - 1)
    corr1, clstrs, silh = cluster_kmeans_base(corr0,
                                              max_num_clusters=max_num_clusters,
                                              min_num_clusters=min_num_clusters,
                                              n_init=n_init, debug=debug)
    # clstrs_tstats = {i:np.mean(silh[clstrs[i]]) / max(np.std(silh[clstrs[i]]), _eps) for i in clstrs.keys()}
    clstrs_tstats = {i:np.mean(silh[clstrs[i]]) for i in clstrs.keys()}
    tstats_mean = np.mean(list(clstrs_tstats.values()))
    redo_clstrs = [i for i in clstrs_tstats.keys() if clstrs_tstats[i] < tstats_mean]
    if len(redo_clstrs) <= 2:
        return corr1, clstrs, silh
    else:
        keys_redo = [j for i in redo_clstrs for j in clstrs[i]]
        corr_tmp = corr0.loc[keys_redo, keys_redo]
        min_num_clusters_ = min_num_clusters - (len(clstrs) - len(redo_clstrs))
        min_num_clusters_ = max(2, min_num_clusters_)
        max_num_clusters_ = min(max_num_clusters, corr_tmp.shape[1] - 1)
        min_num_clusters_ = min(max_num_clusters_, min_num_clusters_)
        corr2, clstrs2, silh2 = cluster_kmeans_base(corr_tmp,
                                                    max_num_clusters=max_num_clusters_,
                                                    min_num_clusters=min_num_clusters_,
                                                    n_init=n_init,
                                                    debug=debug)
        clstrs1 = {i: clstrs[i] for i in clstrs.keys() if i not in redo_clstrs}
        corr_new, clstrs_new, silh_new = make_new_outputs(corr0, clstrs1, clstrs2)
        # new_clstrs_tstats = {i:np.mean(silh_new[i]) / max(np.std(silh_new[i]),  _eps) for i in clstrs_new.keys()}
        new_clstrs_tstats = {i:np.mean(silh_new[i]) for i in clstrs_new.keys()}
        tstats_mean = np.mean(list(clstrs_tstats.values()))
        new_tstats_mean = np.mean(list(new_clstrs_tstats.values()))
        if new_tstats_mean <= tstats_mean:
            return corr1, clstrs, silh
        else:
            return corr_new, clstrs_new, silh_new