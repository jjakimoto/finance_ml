import numpy as np
import pandas as pd
import random


def generateData(nObs, size0, size1, sigma1):
    # Time series of correlated variables
    #1) generating some uncorrelated data
    np.random.seed(seed=12345)
    random.seed(12345)
    x = np.random.normal(0, 1, size=(nObs, size0))  # each row is a variable
    #2) creating correlation between the variables
    cols = [random.randint(0, size0 - 1) for i in range(size1)]
    y = x[:, cols] + np.random.normal(0, sigma1, size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    x = pd.DataFrame(x, columns=range(1, x.shape[1] + 1))
    return x, cols
