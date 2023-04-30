import numpy as np
import pandas as pd



#Randomly generating new 1lvl queue when the previous one is depleted
def OBSecondLevelRand(bids_2lvl, asks_2lvl, horizon = 10, window=100):
    bids_preds = bids_2lvl.rolling(window).apply(lambda x: x.sample(1))
    asks_preds = asks_2lvl.rolling(window).apply(lambda x: x.sample(1))

    return bids_preds.shift(horizon), asks_preds.shift(horizon)



#Order Flow (only for DataFrames, not Series)
def OrderFlow(asks_amounts, bids_amounts):
    updates_asks = pd.DataFrame(index = asks_amounts.index)
    updates_bids = pd.DataFrame(index = bids_amounts.index)

    for i, col in enumerate(asks_amounts.columns):
        updates_asks[f'ask {i} change'] = asks_amounts[col] - asks_amounts[col].shift(1)

    for i, col in enumerate(bids_amounts.columns):
        updates_bids[f'bid {i} change'] = bids_amounts[col] - bids_amounts[col].shift(1)

    return updates_asks, updates_bids



#Heavy Traffic Approximation
def HeavyTrafficApproximation(asks_amounts, bids_amounts, level = 0, t = 10):
    ask_updates, bid_updates = OrderFlow(asks_amounts, bids_amounts)
    ask_updates = ask_updates[f'level {level} change']
    bid_updates = bid_updates[f'level {level} change']
    
    qa_tn = ask_updates.rolling(t).sum()
    qb_tn = bid_updates.rolling(t).sum()
    
    return qa_tn/(t ** 0.5), qb_tn/(t ** 0.5)



#Orders arrival intervals
def OrderIntervals(amounts, time, method = 'datetime'):
    if method == 'datetime':
        differences = time.diff().astype('timedelta64[ns]')
        
        try:
            if amounts.shape[1] != 0:
                isdifferent = (amounts != amounts.shift(1)).sum(axis=1) > 0
        except:
            isdifferent = (amounts != amounts.shift(1))
            
        differences[isdifferent] = pd.Timedelta('0ns')
        differences.fillna('0ns', inplace=True)
        for i in range(1,len(differences)):
            if differences[i] != pd.Timedelta('0ns'):
                differences[i] += differences[i-1]
        
    return differences/pd.Timedelta(seconds=1)



#lambda = N_arrivals / Sum(arrival intervals). Measure of intensity: larger lambda -> more intensive
def extractlambda(amounts, time, window=1000):
    obi = OrderIntervals(amounts, time)
    T = obi[(obi == 0).shift(-1).fillna(False)]
    
    seq = T.rolling(window).mean()

    seq = seq.reindex(index = np.arange(max(seq.index)+1))
    seq.fillna(method='ffill', inplace = True)
    
    lamdba = 1/seq
    lamdba[lamdba == np.inf] = np.nan
    lamdba[lamdba == -np.inf] = np.nan
    
    return lamdba



#correlation of events in ask side and bid side of LOB
def corr_OF(asks_amounts, bids_amounts, window=100, return_vars = False):
    Va, Vb = OrderFlow(asks_amounts, bids_amounts)
    
    ask = Va.sum(axis=1)
    bid = Vb.sum(axis=1)
    
    p = ask.rolling(window).corr(bid)
    
    if return_vars:
        return p, ask.rolling(window).std(), bid.rolling(window).std()
    
    return p



#Spitzer tail
def Spitzer_tail(asks_amounts, bids_amounts, window=100):
    p = corr_OF(asks_amounts, bids_amounts, window)
    
    return np.pi/(np.pi+2*np.arcsin(p))



#Probability of next price to increase. General case.
def P_up(asks_amounts, bids_amounts, time, window=100, window_lambda=10000):
    p, va, vb = corr_OF(asks_amounts, bids_amounts, window, return_vars=True)
    lambda_a = extractlambda(asks_amounts, time, window_lambda)
    lambda_b = extractlambda(bids_amounts, time, window_lambda)
    
    ask = asks_amounts.sum(axis=1)
    bid = bids_amounts.sum(axis=1)
    
    y = ask/(lambda_a**0.5)/va
    x = bid/(lambda_b**0.5)/vb
    
    num = np.arctan(((1+p)/(1-p))**0.5 * (y-x)/(y+x))
    den = 2 * np.arctan(((1+p)/(1-p))**0.5)
    
    return 1/2 - num/den



#Expecancy of next tick price, if corr_OF=-1
def Burghardt(asks_amounts, bids_amounts, ask_lowest_price, bid_highest_price):
    try:
        if asks_amounts.shape[1] != 0:
            share_a = asks_amounts.sum(axis=1)/(asks_amounts.sum(axis=1) + bids_amounts.sum(axis=1))
            share_b = bids_amounts.sum(axis=1)/(asks_amounts.sum(axis=1) + bids_amounts.sum(axis=1))
    except:
        share_a = asks_amounts/(asks_amounts+ bids_amounts)
        share_b = bids_amounts/(asks_amounts+ bids_amounts)
        
    P = share_a*ask_lowest_price + share_b*bid_highest_price
    
    return P



#Price volatility being expressed in terms of OrderFlow
def PriceVolatility_withOF(asks_amounts, bids_amounts, time, basis_point_price, window=1000):
    lamdba = extractlambda(pd.concat([asks_amounts, bids_amounts], axis=1), time)
    updates = pd.concat(OrderFlow(asks_amounts, bids_amounts), axis=1)
    
    v = updates.sum(axis=1).rolling(window).var()
    Df = (asks_amounts[asks_amounts.columns[1]] * bids_amounts[bids_amounts.columns[1]]).rolling(window).mean()
    
    sigma2 = np.pi * (basis_point_price**2) * (v**2) * lamdba / Df
    
    return sigma2 ** 0.5



#Total LOB volume
def Volume(asks_amounts, bids_amounts):
    
    return asks_amounts.sum(axis=1) + bids_amounts.sum(axis=1)



#Spreads
def Spread(asks_prices, bids_prices):
    
    return asks_prices[asks_prices.columns[0]] - bids_prices[bids_prices.columns[0]]



#Liquidity at best price
def BestPriceLiquidity(asks_amounts, bids_amounts):
    
    return asks_amounts[asks_amounts.columns[0]], bids_amounts[bids_amounts.columns[0]]
