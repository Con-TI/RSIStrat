from backtesting import Backtest, Strategy
import pandas_ta as ta
import yfinance as yf
import numpy as np
import pandas as pd

data = yf.download(tickers=['AALI.JK'],period='max').drop(columns=['Adj Close'])

def rsi(open, close, n_rsi):
    diff = np.array(close-open)
    is_gain = diff>0
    is_loss = diff<0
    gain = diff
    loss = -1*diff
    gain[is_loss] = 0
    loss[is_gain] = 0
    gain = pd.Series(gain).rolling(n_rsi).mean()
    loss = pd.Series(loss).rolling(n_rsi).mean()
    rs = gain/loss
    rs = rs.fillna(50)
    rsi = 100-100/(1+rs)
    return rsi.to_numpy()

def lower_bound(close):
    n = len(np.array(close))
    return np.array([10]*n)

class RSIOverBoughtOverSold(Strategy):
    def init(self):
        self.n_rsi = 14
        close = self.data.Close
        open = self.data.Open
        self.rsi = self.I(rsi,open, close,self.n_rsi)        
        self.low_bound = self.I(lower_bound, close)
    def next(self):
        price = self.data.Close[-1]
        if self.rsi<self.low_bound:
            self.buy(tp=price*1.50, sl=price*0.99)

bt = Backtest(data, RSIOverBoughtOverSold, cash=100000000, commission=0.004, exclusive_orders=True)

output = bt.run()
bt.plot()