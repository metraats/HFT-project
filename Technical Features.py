import numpy as np
import pandas as pd


#AccumulationDistributionLine
def ADL(price, amount, window=10, fillnans = True):

    C = price
    L = price.rolling(window).min()
    H = price.rolling(window).max()

    MoneyFlowMultiplier = (2*C-L-H)/(H-L)
    BlockPeriodVolume = price

    MoneyFlowVolumet = MoneyFlowMultiplier * amount.rolling(window).mean()
    ADL = MoneyFlowVolumet.cumsum()

    if fillnans:
        ADL.fillna(method = 'ffill', inplace = True)

    return ADL



#Average directional index
def ADX(price, window=14, fillnans = True):

    L = price.rolling(window).min()
    H = price.rolling(window).max()
    CL = price

    plusM = H - H.shift(1)
    minusM = L.shift(1) - L

    plusMindicator = (plusM >= minusM) & (plusM >= 0) * 1
    minusMindicator = (plusM <= minusM) & (minusM >= 0) * 1

    plusDM = plusM * plusMindicator
    minusDM = minusM * minusMindicator

    TR = pd.concat([H, CL.shift(1)], axis = 1).max(axis = 1) - pd.concat([L, CL.shift(1)], axis = 1).min(axis = 1)


    plusDI = (plusDM/TR).rolling(window-1).mean().shift(1) * (window-1)/window + (plusDM/TR)/window
    minusDI = (minusDM/TR).rolling(window-1).mean().shift(1) * (window-1)/window + (minusDM/TR)/window

    ADX = 100 * (abs(plusDI - minusDI)/(plusDI + minusDI)).rolling(window-1).mean().shift(1)*(window-1)/window + 100*(abs(plusDI - minusDI)/(plusDI + minusDI))/window

    if fillnans:
        ADX.fillna(method = 'ffill', inplace = True)

    return ADX



#Chande Momentum oscillator
def CMO(price, window=10, fillnans = True):
    updays = (price / (price.shift(1)) > 1) * 1
    downdays = (price / (price.shift(1)) < 1) * 1
    zerodays = (price / (price.shift(1)) == 1) * 1

    CL_up = (price - (price.shift(1))) * updays
    CL_down = -(price - (price.shift(1))) * downdays

    Su = CL_up.rolling(window).sum()
    Sd = CL_down.rolling(window).sum()

    CMO = 100 * (Su - Sd)/(Su + Sd)

    if fillnans:
        CMO.fillna(method = 'ffill', inplace = True)

    return CMO



#Momentum
def Momentum(price):
    MOM = price - price.shift(1)
    
    return MOM



#Rate of Change
def ROC(price, window=10):
    ROC = (price - price.shift(window)) / price.shift(window) * 100
    
    return ROC



#Relative Strength Index
def RSI(price, window=140, fillnans = True):
    CL = price - price.shift(1)

    updays = ((CL - CL.shift(1)) > 0) * 1
    downdays = ((CL - CL.shift(1)) < 0) * 1
    zerodays = ((CL - CL.shift(1)) == 0) * 1

    CL_up = CL * updays
    CL_down = -CL * downdays

    AG = CL_up.rolling(window).sum()
    AL = CL_down.rolling(window).sum()

    Relative_strength = AG/AL

    RSI = 100 - 100/(1 + Relative_strength)

    if fillnans:
        RSI.fillna(method = 'ffill', inplace = True)

    return RSI



#Stochastic Relative Strength Index
def StochasticRSI(price, window_stochastic=140, window_rsi=140, fillnans = True):
    rsi = RSI(price, window_rsi)

    RSI_max = rsi.rolling(window_stochastic).max()
    RSI_min = rsi.rolling(window_stochastic).min()

    StochRSI = (rsi - RSI_min)/(RSI_max - RSI_min)

    if fillnans:
        StochRSI.fillna(method = 'ffill', inplace = True)

    return StochRSI



#Linear regression line
def LinearRegressionLine(price, window_learning=10, window_forecasting=1, fillnans = True):
    X_train = price
    y_train = price.shift(-window_forecasting)

    X_mean = X_train.rolling(window_learning).mean()
    y_mean = y_train.rolling(window_learning).mean()

    XminusX_mean = X_train - X_mean
    YminusY_mean = y_train - y_mean

    b1 = (XminusX_mean * YminusY_mean).rolling(window_learning).sum() / (XminusX_mean * XminusX_mean).rolling(window_learning).sum()#ddd
    b1[b1 == np.inf] = np.nan
    b1[b1 == -np.inf] = np.nan
    b0 = y_mean - b1 * X_mean

    if fillnans:
        b1.fillna(method = 'ffill', inplace = True)
        b0.fillna(method = 'ffill', inplace = True)

    return b0, b1
