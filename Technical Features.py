import numpy as np
import pandas as pd


#AccumulationDistributionLine
def ADL(price, amount, block=10, fillnans = True):
    C = price.copy()
    L = price.rolling(block).min()
    H = price.rolling(block).max()

    MoneyFlowMultiplier = (2*C-L-H)/(H-L)
    BlockPeriodVolume = price

    MoneyFlowVolumet = MoneyFlowMultiplier * amount.rolling(block).mean()
    ADL = MoneyFlowVolumet.cumsum()

    if fillnans:
        ADL.fillna(method = 'ffill', inplace = True)

    return ADL



#Average directional index
def ADX(price, block = 10, window=14, fillnans = True):
    L = price.rolling(block).min()
    H = price.rolling(block).max()
    CL = price.copy()

    plusM = H - H.shift(1)
    minusM = L.shift(1) - L

    plusMindicator = (plusM >= minusM) & (plusM >= 0) * 1
    minusMindicator = (plusM <= minusM) & (minusM >= 0) * 1

    plusDM = plusM * plusMindicator
    minusDM = minusM * minusMindicator

    TR = pd.concat([H, CL.shift(1)], axis = 1).max(axis = 1) - pd.concat([L, CL.shift(1)], axis = 1).min(axis = 1)

    plusDI = (plusDM/TR).rolling(block*window-1).mean().shift(1) * (window*block-1)/window + (plusDM/TR)/window
    minusDI = (minusDM/TR).rolling(block*window-1).mean().shift(1) * (block*window-1)/window + (minusDM/TR)/window

    ADX = 100 * (abs(plusDI - minusDI)/(plusDI + minusDI)).rolling(block*window-1).mean().shift(1)*(block*window-1)/window + 100*(abs(plusDI - minusDI)/(plusDI + minusDI))/window

    if fillnans:
        ADX.fillna(method = 'ffill', inplace = True)

    return ADX



#Chande Momentum oscillator
def CMO(price, block=10, window=19, fillnans = True):
    updays = (price / (price.shift(1)) > 1) * 1
    downdays = (price / (price.shift(1)) < 1) * 1
    zerodays = (price / (price.shift(1)) == 1) * 1

    CL_up = (price - (price.shift(1))) * updays
    CL_down = -(price - (price.shift(1))) * downdays

    Su = CL_up.rolling(window*block).sum()
    Sd = CL_down.rolling(window*block).sum()

    CMO = 100 * (Su - Sd)/(Su + Sd)

    if fillnans:
        CMO.fillna(method = 'ffill', inplace = True)

    return CMO



#Momentum
def Momentum(price):
    MOM = price - price.shift(1)
    
    return MOM



#Rate of Change
def ROC(price, block=10, window=12):
    ROC = (price - price.shift(block*window)) / price.shift(block*window) * 100
    
    return ROC



#Relative Strength Index
def RSI(price, block=10, window=14, fillnans = True):
    CL = price - price.shift(1)

    updays = ((CL - CL.shift(1)) > 0) * 1
    downdays = ((CL - CL.shift(1)) < 0) * 1
    zerodays = ((CL - CL.shift(1)) == 0) * 1

    CL_up = CL * updays
    CL_down = -CL * downdays

    AG = CL_up.rolling(block*window).sum()
    AL = CL_down.rolling(block*window).sum()

    Relative_strength = AG/AL

    RSI = 100 - 100/(1 + Relative_strength)

    if fillnans:
        RSI.fillna(method = 'ffill', inplace = True)

    return RSI



#Stochastic Relative Strength Index
def StochasticRSI(price, block=10, window=14, rsi_block = 10, fillnans = True):
    rsi = RSI(price, block=10, window=14)

    RSI_max = rsi.rolling(block*window*rsi_block).max()
    RSI_min = rsi.rolling(block*window*rsi_block).min()

    StochRSI = (rsi - RSI_min)/(RSI_max - RSI_min)

    if fillnans:
        StochRSI.fillna(method = 'ffill', inplace = True)

    return StochRSI



#Linear regression line
def LinearRegressionLine(price, window_learning=10, window_forecasting=1, fillnans = True):
    X_train = price.copy()
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



#Realized Volatility
def RealizedVolatility(price, block=10, window=10, fillnans = True):
    v = (price - price.shift(1)) * (price - price.shift(1))
    Realized_Volatility = v.rolling(block*window).mean()
    
    if fillnans:
        Realized_Volatility.fillna(method = 'ffill', inplace = True)
        
    return Realized_Volatility



#Realized Kernel
def RealizedKernel(price, horizon=7, window=10, kernel = 'Bartlett'):
    def k(x, kernel):
        if kernel == 'Bartlett':
            return 1-x
        
    returns = price - price.shift(1)
    
    shifts_covs = pd.DataFrame(index=price.index)
    for i in range(1,horizon+1):
        shifts_covs[f'h={i}'] = returns.rolling(window).cov(returns.shift(i)) * window * k(x=(i-1)/horizon, kernel=kernel)
    for i in range(-horizon,0):
        shifts_covs[f'h={i}'] = returns.rolling(window).cov(returns.shift(i)) * window * k(x=(-i-1)/horizon, kernel=kernel)
    
    variance = price.rolling(window).var()
    K = variance + shifts_covs.sum(axis=1)
    
    return K



#Realized Bipower Variation
def RealizedBipowerVariation(price, block=10, window=10, fillnans = True):
    delta = (price-price.shift(1)) * (price-price.shift(1)).shift(1)
    t = delta.rolling(block*window).mean()
    
    RealizedBipowerVolatility = t * np.pi * 0.5
    
    if fillnans:
        RealizedBipowerVolatility.fillna(method = 'ffill', inplace = True)
        
    return RealizedBipowerVolatility



#Jump Variation
def JumpVariation(price, block=10, window=10, fillnans = True):
    JumpVariation = RealizedVolatility(price, block=block, window=window, fillnans = fillnans) - RealizedBipowerVariation(price, block=block, window=window, fillnans = fillnans)
    
    return JumpVariation



#Past Returns
def PastReturns(price):
    PRET = price.pct_change() * 10000
    PRET.mean()
    
    return PRET



#Mean Divirgence
def MeanDivirgence(prices_1, prices_2, block=10, window = 10, fillnans = True):
    d = prices_1/prices_2 - 1
    d[d == np.inf] = np.nan
    d[d == -np.inf] = np.nan
    
    DIV = d - d.rolling(window*block).mean()
    
    if fillnans:
        DIV.fillna(method = 'ffill', inplace = True)
    
    return DIV



#Awesome Oscillator
def AwesomeOscillator(price, block = 10, window_big = 34, window_small = 5):
    H = price.rolling(block).max()
    L = price.rolling(block).min()
    
    AO = (H+L).rolling(window_small*block).mean()/2 - (H+L).rolling(window_big*block).mean()/2
    
    return AO



#Accelerator Oscillator
def AcceleratorOscillator(price, AO_window = 5, block = 10, window_big = 34, window_small = 5):
    AO = AwesomeOscillator(price=price, block = block, window_big = window_big, window_small = window_small)
    
    return AO - AO.rolling(AO_window).mean()



#Average directional movement index rating
def ADXR(price, block=10, window = 14, fillnans = True):
    adx = ADX(price=price, block=block, window=window, fillnans = fillnans)
    
    return (adx + adx.shift(1))/2



#Williams Alligator Indicator
def AlligatorIndicator(price, block=10, w_jaw=13, w_teeth=8, w_lips=5):
    H = price.rolling(block).max()
    L = price.rolling(block).min()
    
    jaw = (H+L).rolling(block*w_jaw).mean()/2
    teeth = (H+L).rolling(block*w_teeth).mean()/2
    lips = (H+L).rolling(block*w_lips).mean()/2
    
    return jaw, teeth, lips



#Absolute price oscillator
def APO(price, block=10, small_window=5, big_window=13):
    H = price.rolling(block).max()
    L = price.rolling(block).min()
    
    M = (H+L)/2
    APO = M.ewm(small_window*block).mean() - M.ewm(big_window*block).mean()
    
    return APO



#Average True Range
def ATR(price, block=10, window=14, Normalized = False):
    L = price.rolling(block*window).min()
    H = price.rolling(block*window).max()
    CL = price.copy()
    
    TR = pd.concat([H-L, abs(H-CL.shift(1)), abs(L-CL.shift(1))], axis = 1).max(axis = 1)
    ATR = TR.ewm(alpha = 1/(block*window), adjust=False).mean()
    
    if Normalized:
        return ATR/CL*100
    
    return ATR



#Bollinger Bands
def Bollinger(price, block=10, window=20, k=2):
    
    BB_std = price.rolling(block*window).std()
    BB_middle = price.rolling(block*window).mean()
    BB_lower = BB_middle - k*BB_std
    BB_upper = BB_middle + k*BB_std
    
    return BB_middle, BB_upper, BB_lower



#Ichimoku clouds
def IchimokuClouds(price, block=10, window_convline=9, window_baseline=26, window_leadingspanb=52):
    
    ConversionLine = (price.rolling(block*window_convline).max() + price.rolling(block*window_convline).min())/2
    Baseline = (price.rolling(block*window_baseline).max() + price.rolling(block*window_baseline).min())/2
    LeadingSpanA = (ConversionLine+Baseline)/2
    LeadingSpanB = (price.rolling(block*window_leadingspanb).max() + price.rolling(block*window_leadingspanb).min())/2
    LaggingSpan = price.shift(block*window_baseline)
    
    return ConversionLine, Baseline, LeadingSpanA, LeadingSpanB, LaggingSpan



#Chaikin Oscillator
def ChaikinOscillator(price, volumes, block=10, ema1 = 3, ema2 = 10):
    
    L = price.rolling(block).min()
    H = price.rolling(block).max()
    V = volumes.rolling(block).sum()
    
    N = (2*price - L - H)/(H - L)
    M = N * V
    ADL = M + M.shift(1)
    CO = ADL.ewm(ema1).mean() - ADL.ewm(ema2).mean()
    
    return CO



#Chandelier Exit
def ChandelierExit(price, block=10, window=22, k=3):
    
    L = price.rolling(block*window).min()
    H = price.rolling(block*window).max()
    
    ChandelierLong = H - k * ATR(price, block=block, window=window)
    ChandelierShort = L + k * ATR(price, block=block, window=window)
    
    return ChandelierLong, ChandelierShort



#Center of gravity
def COG(price, block = 10, n = 1):
    W = price.copy()
    
    for i in range(1, block*n):
        W += price.shift(i) * (i+1)
        
    return - W / price.rolling(block*n).sum()



#Double exponential moving average
def DEMA(price, block=10, window=20):
    L = price.rolling(block).min()
    H = price.rolling(block).max()
    M = (H+L)/2
    
    DEMA = 2 * M.ewm(block*window).mean() - M.ewm(block*window).mean().ewm(block*window).mean()
    return DEMA



#Detrended price oscillator
def DPO(price, block=10, window=20):
    A = price.shift(block*window//2 + 1)
    B = price.rolling(block*window).mean()
    
    return A-B



#Weighted Moving Average
def WMA(price, block = 10, window = 10):    
    W = price.copy() * block*window
    d = 1
    
    for i in range(1, block*window):
        W += price.shift(i) * (block*window - i)
        d += (i+1)
        
    return W/d



#Hull Moving Average
def HullMA(price, block=10, window=10):
    n = block * window
    WMA1 = WMA(price.shift(n//2), block=block, window = window)
    WMA2 = WMA(price.shift(n), block=block, window = window)
    
    HMA = WMA(price = 2*WMA1-WMA2, block = int(np.sqrt(block)), window = int(np.sqrt(window)))
    
    return HMA



#Internal bar strength
def IBS(price, block=10):
    
    L = price.rolling(block).min()
    H = price.rolling(block).max()
    
    return (price - L)/(H - L)



#Percentage price oscillator
def PPO(price, block=10, small_window=12, big_window=26):
    
    MACD = price.ewm(small_window*block).mean() - price.ewm(big_window*block).mean()
    PPO = MACD/price.ewm(big_window*block).mean()*100
    
    return PPO

