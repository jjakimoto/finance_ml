import pandas as pd
import talib


def macd_side(close):
    macd, signal, hist = talib.MACD(close.values)
    hist = pd.Series(hist).fillna(1).values
    return pd.Series(2 * ((hist > 0).astype(float) - 0.5),
                     index=close.index[-len(hist):])
