from sklearn.datasets import make_classification
import pandas as pd


def get_cls_data(n_features=40, n_informative=10, n_redundant=10, n_samples=10000):
    X, cont = make_classification(n_samples=n_samples, n_features=n_features,
                                  n_informative=n_informative, n_redundant=n_redundant,
                                  random_state=0, shuffle=False)
    time_idx = pd.DatetimeIndex(periods=n_samples, freq=pd.tseries.offsets.BDay(),
                                end=pd.datetime.today())
    X = pd.DataFrame(X, index=time_idx)
    cont = pd.Series(cont, index=time_idx).to_frame('bin')
    # Create name of columns
    columns = ['I_' + str(i) for i in range(n_informative)]
    columns += ['R_' + str(i) for i in range(n_redundant)]
    columns += ['N_' + str(i) for i in range(n_features - len(columns))]
    X.columns = columns
    cont['w'] = 1. / cont.shape[0]
    cont['t1'] = pd.Series(cont.index, index=cont.index)
    return X, cont