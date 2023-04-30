import numpy as np
import pandas as pd

def OBSecondLevelRand(bids_2lvl, asks_2lvl, horizon = 10, window=100):
    bids_preds = bids_2lvl.rolling(window).apply(lambda x: x.sample(1))
    asks_preds = asks_2lvl.rolling(window).apply(lambda x: x.sample(1))

    return bids_preds.shift(horizon), asks_preds.shift(horizon)


def OBupdates(asks_amounts, bids_amounts):
    updates_asks = pd.DataFrame(index = asks_amounts.index)
    updates_bids = pd.DataFrame(index = bids_amounts.index)

    for i, col in enumerate(asks_amounts.columns):
        updates_asks[f'ask {i} change'] = asks_amounts[col] - asks_amounts[col].shift(1)

    for i, col in enumerate(bids_amounts.columns):
        updates_bids[f'bid {i} change'] = bids_amounts[col] - bids_amounts[col].shift(1)

    return updates_asks, updates_bids