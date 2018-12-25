import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, f1_score, recall_score, precision_score,\
    precision_recall_curve, roc_curve



from finance_ml.multiprocessing import mp_pandas_obj


def mp_train_times(train_times, test_times, molecule):
    trn = train_times[molecule].copy(deep=True)
    for init, end in test_times.iteritems():
        df0 = trn[(init <= trn.index) & (trn.index <= end)].index
        df1 = trn[(init <= trn) & (trn <= end)].index
        df2 = trn[(trn.index <= init) & (end <= trn)].index
        trn = trn.drop(df0 | df1 | df2)
    return trn


def get_train_times(train_times, test_times, num_threads=1):
    """Sample train points without overlapping with test period
    
    Params
    ------
    train_times: pd.Series
        Trainig points with index for initial and values for end time
    test_times: pd.Series
        Testing points with index for initial and values for end time
    num_threads: int, default 1
        The number of thrads for multiprocessing
        
    Returns
    -------
    pd.Series
    """
    return mp_pandas_obj(mp_train_times, ('molecule', train_times.index), num_threads,
                         train_times=train_times, test_times=test_times)


def get_embargo_times(times, pct_embargo):
    """Get embargo time index for each timestamp
    
    times:
        times: Timestamps
            Entire timestamps which you want to apply embargo
        pct_embargo: float ranged at [0, 1]
            The ratio to embargo with respect to the size of timestamps
            
    Returns:
        pd.Series: For each valud corresponds to a point which you should take
        out before from the other forward dataset
    """
    step = int(times.shape[0] * pct_embargo)
    if step == 0:
        embg = pd.Series(times, index=times)
    else:
        embg = pd.Series(times[step:], index=times[:-step])
        embg = embg.append(pd.Series(times[-1], index=times[-step:]))
    return embg


def evaluate(model, X, y, method, sample_weight=None, labels=None, pos_idx=1, pos_label=1):
    """Calculate score
    
    Params
    ------
    model: Trained classifier instance
    X: array-like, Input feature
    y: array-like, Label
    method: str
        The name of scoring methods. 'precision', 'recall', 'f1', 'precision_recall',
        'roc', 'accuracy' or 'neg_log_loss'
    sample_weight: pd.Series, optional
        If specified, apply this to bot testing and training
    labels: array-like, optional
        The name of labels
        
    Returns
    -------
    list of scores
    """
    if method == 'f1':
        pred = model.predict(X)
        score = f1_score(y, pred, sample_weight=sample_weight, labels=labels)
    elif method == 'neg_log_loss':
        prob = model.predict_proba(X)
        score = -log_loss(y, prob, sample_weight=sample_weight, labels=labels)
    elif method == 'precision':
        pred = model.predict(X)
        score = precision_score(y, pred, pos_label=pos_label, sample_weight=sample_weight)
    elif method == 'recall':
        pred = model.predict(X)
        score = recall_score(y, pred, pos_label=pos_label, sample_weight=sample_weight)
    elif method == 'precision_recall':
        prob = model.predict_proba(X)[:, pos_idx]
        score = precision_recall_curve(y, prob, pos_label=pos_label, sample_weight=sample_weight)
    elif method == 'roc':
        prob = model.predict_proba(X)[:, pos_idx]
        score = roc_curve(y, prob, pos_label=pos_label, sample_weight=sample_weight)
    elif method == 'accuracy':
        pred = model.predict(X)
        score = accuracy_score(y, pred, sample_weight=sample_weight)
    else:
        raise Exception(f'No Implementation method={method}')
    return score