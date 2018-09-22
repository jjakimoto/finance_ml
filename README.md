# finance_ml
Python implementations of Machine Learning helper functions based on a book,
`Advances in Financial Machine Learning`[[1]](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089),
written by `Marcos Lopez de Prado`. 

# Installation
Excute the following command
```python
python setup.py install
```

# Implementation
## labeling
* Triple Barriers Labeling
* CUSUM sampling
```Python
from finance_ml.labeling import get_barrier_labels, cusum_filter
from finance_ml.stats import get_daily_vol

vol = get_daily_vol(close)
trgt = vol
timestamps = cusum_filter(close, vol)
labels = get_barrier_labels(close, timestamps, trgt, sltp=[1, 1],
                            num_days=1, min_ret=0, num_threads=16)
print(labels.show())
```
Return the following pandas.Series
```python
2000-01-05 -1.0
2000-01-06  1.0
2000-01-10 -1.0
2000-01-11  1.0
2000-01-12  1.0
```
* Future Returns for Regression

## multiprocessing
Parallel computing using `multiprocessing` library.
Here is the example of applying function to each element with parallelization.
```python
import pandas as pd
import numpy as np

def apply_func(x):
    return x ** 2

def func(df, timestamps, f):
    df_ = df.loc[timestamps]
    for idx, x in df_.items():
        df_.loc[idx] = f(x)
    return df_
    
df = pd.Series(np.random.randn(10000))
from finance_ml.multiprocessing import mp_pandas_obj

results = mp_pandas_obj(func, pd_obj=('timestamps', df.index),
                        num_threads=24, df=df, f=apply_func)
print(results.head())
```
Output:
```
0    0.449278
1    1.411846
2    0.157630
3    4.949410
4    0.601459
dtype: float64
```
