# finance_ml
Python implementations of Machine Learning helper functions for Quantiative Finance based on books,
[Advances in Financial Machine Learning](https://www.amazon.co.jp/Advances-Financial-Machine-Learning-English-ebook/dp/B079KLDW21) and [Machine Learning for Asset Managers](https://www.amazon.com/Machine-Learning-Managers-Elements-Quantitative/dp/1108792898) , written by `Marcos Lopez de Prado`. 

# Installation
Excute the following command
```python
python setup.py install
```

or

Simply add `your/path/to/finace_ml` to your PYTHONPATH.

# Implementation
The following functions are implemented:
* Labeling
* Multiporcessing
* Sampling
* Feature Selection
* Asset Allcation
* Breakout Detection

# Examples
Some of example notebooks are found under the folder `MLAssetManagers`.

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
```

For more detail, please refer to example notebook!