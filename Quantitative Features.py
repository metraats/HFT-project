import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


#Order Book Imbalance
def OBI(asks_amounts, bids_amounts):

    Va = asks_amounts.mean(axis = 1)
    Vb = bids_amounts.mean(axis = 1)

    OBI = (Vb-Va)/(Va+Vb)
    return OBI



#Autocorrelation (can be used not only for price time series)
def Autocorrelation(price, window=1000, lag=10, fillnans = True, delete_outliers=True):
    shifted = price.shift(lag)
    correlation = price.rolling(window).corr(shifted)

    if delete_outliers:
        correlation[abs(correlation) > 1] = np.nan

    if fillnans:
        correlation.fillna(method = 'ffill', inplace = True)

    return correlation



#Partial Correlation (can be used not only for price time series)
def PartialCorrelation(price, window=1000, lag=10, fillnans = True):

    prices_copy = price.copy()
    answ = pd.DataFrame(index = price.index)

    for i in range(1, lag + 1):
        shifted = prices_copy.shift(i)
        correlation = prices_copy.rolling(window).corr(shifted)

        if fillnans:
            correlation.fillna(method = 'ffill', inplace = True)

        answ[f'PACF for lag {i}'] = correlation
        prices_copy -= correlation * shifted

    return answ



#Cointegration
def Cointegration(asks_amounts, bids_amounts, window=10, fillnans=True, method = 'total volume', depth = 25):

    if method == 'by level':

        b1 = pd.DataFrame(index = ASK.index)
        b0 = pd.DataFrame(index = ASK.index)

        for i in range(depth):
            y = asks_amounts
            X = bids_amounts

            X_mean = X.rolling(window).mean()
            y_mean = y.rolling(window).mean()

            XminusX_mean = X - X_mean
            YminusY_mean = y - y_mean

            b_1 = (XminusX_mean * YminusY_mean).rolling(window).sum() / (XminusX_mean * XminusX_mean).rolling(window).sum()#ddd
            b_1[b_1 == np.inf] = np.nan
            b_1[b_1 == -np.inf] = np.nan
            b_0 = y_mean - b_1 * X_mean

            b1[f'LVL {i}'] = b_1
            b0[f'LVL {i}'] = b_0

    elif method == 'total volume':
        y = asks_amounts.sum(axis = 1)
        X = bids_amounts.sum(axis = 1)

        X_mean = X.rolling(window).mean()
        y_mean = y.rolling(window).mean()

        XminusX_mean = X - X_mean
        YminusY_mean = y - y_mean

        b1 = (XminusX_mean * YminusY_mean).rolling(window).sum() / (XminusX_mean * XminusX_mean).rolling(window).sum()#ddd
        b1[b1 == np.inf] = np.nan
        b1[b1 == -np.inf] = np.nan
        b0 = y_mean - b1 * X_mean

    return b0, b1



#Adaptive Logistic Regression
def AdaptiveLogreg(asks_amounts=None, bids_amounts=None, ask_lowest_price=None, bid_highest_price=None,
                   window=10, lag=1, method='binary', side = None, depth = 6):
    #Сам логрег
    def logregfunc(X, xtest, y, method='binary'):
        lr = LogisticRegression(solver = 'newton-cg')
        lr.fit(X,y)

        if method=='triple':
            if set(y.unique()) == set([-1, 1]):
                ret = np.array([lr.predict_proba(xtest)[0][0], 0, lr.predict_proba(xtest)[0][1]])
            elif set(y.unique()) == set([0, 1]):
                ret = np.array([0, lr.predict_proba(xtest)[0][0], lr.predict_proba(xtest)[0][1]])
            elif set(y.unique()) == set([-1, 0]):
                ret = np.array([lr.predict_proba(xtest)[0][0], lr.predict_proba(xtest)[0][1], 0])
            else:
                ret = np.array([lr.predict_proba(xtest)[0][0], lr.predict_proba(xtest)[0][1], lr.predict_proba(xtest)[0][2]])

        if method=='binary':
            ret = np.array([lr.predict_proba(xtest)[0][0], lr.predict_proba(xtest)[0][1]])

        return ret

    # Для аска
    if side == 'ask':
        ASK = asks_amounts
        ASK.fillna(0, inplace=True)

        # Изменение/не изменение
        if method == 'binary':
            tgts = (ask_lowest_price != ask_lowest_price.shift(lag)) * 1
            tgts.fillna(0, inplace=True)
            answers = []

            for i in range(window, ASK.shape[0]-1):
                X = ASK[i-window:i]
                y = tgts[i-window:i]
                xtest = ASK.iloc[i:i+1]

                if sum(abs(y)) == 0:
                    preds = np.array([1,0])
                elif sum(y) == window:
                    preds = np.array([0,1])
                else:
                    preds = logregfunc(X, xtest, y, method='binary')
                answers.append(preds)
            answ = pd.DataFrame(answers, columns=['Ask price=','Ask price+-'])

        # 0/рост/падние
        if method == 'triple':
            tgts = ((ask_lowest_price > ask_lowest_price.shift(lag)) * 1) + ((ask_lowest_price < ask_lowest_price.shift(lag)) * -1)
            tgts.fillna(0, inplace=True)
            answers = []

            for i in range(window, ASK.shape[0]-1):
                X = ASK[i-window:i]
                y = tgts[i-window:i]
                xtest = ASK.iloc[i:i+1]

                if sum(abs(y)) == 0:
                    preds = np.array([0,1,0])
                elif sum(y) == window:
                    preds = np.array([0,0,1])
                elif sum(y) == -window:
                    preds = np.array([1,0,0])
                else:
                    preds = logregfunc(X, xtest, y, method='triple')
                answers.append(preds)
            answ = pd.DataFrame(answers, columns=['Ask price-', 'Ask price=', 'Ask price+'])

    # Для бида
    if side == 'bid':
        BIDS = bids_amounts
        BIDS.fillna(0, inplace=True)

        #Изменение/не изменение
        if method == 'binary':
            tgts = (bid_highest_price != bid_highest_price.shift(lag)) * 1
            tgts.fillna(0, inplace=True)
            answers = []

            for i in range(window, BIDS.shape[0]-1):
                X = BIDS[i-window:i]
                y = tgts[i-window:i]
                xtest = BIDS.iloc[i:i+1]

                if sum(abs(y)) == 0:
                    preds = np.array([1,0])
                elif sum(y) == window:
                    preds = np.array([0,1])
                else:
                    preds = logregfunc(X, xtest, y, method='binary')
                answers.append(preds)
            answ = pd.DataFrame(answers, columns=['Bid price=','Bid price+-'])

        # 0/рост/падние
        if method == 'triple':
            tgts = ((bid_highest_price > bid_highest_price.shift(lag)) * 1) + ((bid_highest_price < bid_highest_price.shift(lag)) * -1)
            tgts.fillna(0, inplace=True)
            answers = []

            for i in range(window, BIDS.shape[0]-1):
                X = BIDS[i-window:i]
                y = tgts[i-window:i]
                xtest = BIDS.iloc[i:i+1]

                if sum(abs(y)) == 0:
                    preds = np.array([0,1,0])
                elif sum(y) == window:
                    preds = np.array([0,0,1])
                elif sum(y) == -window:
                    preds = np.array([1,0,0])
                else:
                    preds = logregfunc(X, xtest, y, method='triple')
                answers.append(preds)
            answ = pd.DataFrame(answers, columns=['Bid price-','Bid price=','Bid price+'])
    return answ



#Order Book Internal Imbalance median (For ask and bid sides)
def Internal_imbalance_median_ask(asks_amounts, asks_prices):
    
    median_amounts = asks_amounts.median(axis = 1)
    residuals = median_amounts
    proportions = pd.DataFrame(index = asks_amounts.index, columns = asks_amounts.columns)
    
    for col in asks_amounts.columns:
        proportions[col] = pd.DataFrame([residuals, asks_amounts[col]]).min()
        residuals = residuals - proportions[col] 
    
    Pn = 0
    for col_amount, col_price in list(zip(proportions.columns, asks_prices.columns)):
        Pn += proportions[col_amount] * asks_prices[col_price]
    
    lowest_ask_price = asks_prices[asks_prices.columns[0]]
    IMB_a = (Pn/lowest_ask_price - 1) * 10000
    
    return IMB_a



def Internal_imbalance_median_bid(bids_amounts, bids_prices):
    
    median_amounts = bids_amounts.median(axis = 1)
    residuals = median_amounts
    proportions = pd.DataFrame(index = bids_amounts.index, columns = bids_amounts.columns)
    
    for col in bids_amounts.columns:
        proportions[col] = pd.DataFrame([residuals, bids_amounts[col]]).min()
        residuals = residuals - proportions[col] 
    
    Pn = 0
    for col_amount, col_price in list(zip(proportions.columns, bids_prices.columns)):
        Pn += proportions[col_amount] * bids_prices[col_price]
    
    highest_bid_price = bids_prices[bids_prices.columns[0]]
    IMB_b = (highest_bid_price/Pn - 1) * 10000
    
    return IMB_b



#Trade Flow Imbalance
def TFI(asks_amounts, bids_amounts):
    
    Buy_volume = bids_amounts.sum(axis = 1)
    Sell_volume = asks_amounts.sum(axis = 1)
    TFI = Buy_volume - Sell_volume
    
    return TFI
