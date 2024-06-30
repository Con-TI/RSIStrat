from backtesting import Backtest, Strategy
import yfinance as yf
import numpy as np
import pandas as pd

data = yf.download(tickers=['AALI.JK'],period='max').drop(columns=['Adj Close'])

def rsi(open, close, n_rsis:np.array):
    diff = np.array(close-open)
    is_gain = diff>0
    is_loss = diff<0
    gain = diff
    loss = -1*diff
    gain[is_loss] = 0
    loss[is_gain] = 0
    
    unique_n_rsi = np.unique(n_rsis).tolist()
    gains = {n:pd.Series(gain).rolling(n).mean() for n in unique_n_rsi}
    losses = {n:pd.Series(loss).rolling(n).mean()  for n in unique_n_rsi}
    rss = {n:(gains[n]/losses[n]).fillna(50) for n in unique_n_rsi}
    rsis = {n:100-100/(1+rss[n]) for n in unique_n_rsi}

    return_array = np.zeros(len(diff))

    for idx in range(len(n_rsis)):
        num = n_rsis[idx]
        rsi_val = rsis[num].iloc[idx]
        return_array[idx] = rsi_val

    return return_array

def n_rsis(close):
    num = len(np.array(close))
    return np.random.uniform(10,20,num).astype(int)

def rsi_bounds(close):
    num = len(np.array(close))
    return np.random.uniform(10,20,num)


class RSIOverBoughtOverSold(Strategy):
    def init(self):
        close = self.data.Close
        open = self.data.Open
        high = self.data.High
        low = self.data.Low
        vol = self.data.Volume
        daily_turnover = self.data.Volume * self.data.Open
        nrsis_vals = n_rsis(close)
        self.bounds_vals = self.I(rsi_bounds,close)
        self.rsi = self.I(rsi,open,close,nrsis_vals)

    def next(self):
        price = self.data.Close[-1]
        if self.rsi<self.bounds_vals:
            self.buy(tp=price*1.50, sl=price*0.99)

bt1 = Backtest(data, RSIOverBoughtOverSold, cash=100000000, commission=0.004, exclusive_orders=True)
output = bt1.run()
bt2 = Backtest(data, RSIOverBoughtOverSold, cash=100000000, commission=0.004, exclusive_orders=True)
output2 = bt2.run()
# print(output.loc["Return [%]"])
bt1.plot()
