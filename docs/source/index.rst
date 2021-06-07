.. finance_ml documentation master file, created by
   sphinx-quickstart on Sat Dec 28 14:57:57 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ``finance_ml``'s documentation!
===============================================
Python implementations of Machine Learning helper functions for Quantiative Finance based on a book,
`Advances in Financial Machine Learning`_, written by ``Marcos Lopez de Prado``. 

.. _Advances in Financial Machine Learning: https://www.amazon.co.jp/Advances-Financial-Machine-Learning-English-ebook/dp/B079KLDW21


Installation
--------------
Excute the following command ::

    python setup.py install

Examples
--------------
labeling
~~~~~~~~~
Triple Barriers Labeling and CUSUM sampling::

    from finance_ml.labeling import get_barrier_labels, cusum_filter
    from finance_ml.stats import get_daily_vol

    vol = get_daily_vol(close)
    trgt = vol
    timestamps = cusum_filter(close, vol)
    labels = get_barrier_labels(close, timestamps, trgt, sltp=[1, 1],
                            num_days=1, min_ret=0, num_threads=16)
    print(labels.show())

Return the following pandas.Series::

    2000-01-05 -1.0
    2000-01-06  1.0
    2000-01-10 -1.0
    2000-01-11  1.0
    2000-01-12  1.0

multiprocessing
~~~~~~~~~~~~~~~~
Parallel computing using ``multiprocessing`` library.
Here is the example of applying function to each element with parallelization.::

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

Output::

    0    0.449278
    1    1.411846
    2    0.157630
    3    4.949410
    4    0.601459


Documentation for the Code
============================
.. toctree::
   :maxdepth: 2
   :caption: Contents:

Labeling
---------
.. automodule:: finance_ml.labeling.barriers
    :members:

.. automodule:: finance_ml.labeling.sampling
    :members:

.. automodule:: finance_ml.labeling.sides
    :members:

.. automodule:: finance_ml.labeling.sizes
    :members:

.. automodule:: finance_ml.labeling.utils
    :members:

Multiprocessing
------------------
.. automodule:: finance_ml.multiprocessing.pandas
    :members:

.. automodule:: finance_ml.multiprocessing.partition
    :members:

.. automodule:: finance_ml.multiprocessing.utils
    :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
