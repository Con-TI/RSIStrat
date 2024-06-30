import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


# data = yf.download(tickers=['BBCA.JK','BMRI.JK'],start='2020-01-01', end='2023-12-31')
data = pd.read_pickle('data.pkl')
data = data[(data.index > '2015-01-01') & (data.index < '2023-12-31')]
data = data.loc[:,data.columns.get_level_values(1).isin(['BBCA.JK','AALI.JK'])].dropna()

class BackTest():
    def __init__(self,data):
        self.data = data
        self.close = torch.transpose(torch.tensor(data['Close'].values, device=device),0,1)
        self.open = torch.transpose(torch.tensor(data['Open'].values, device=device),0,1)
        self.changes = self.close - self.open
        self.wealth = torch.ones(self.close.size(),device=device)
        self.years = torch.tensor(data.index.year.to_numpy(), device=device).repeat(self.close.size()[0],1)

    def reset(self):
        self.wealth = torch.ones(self.close.size(),device=device)

    def rsi_calc(self,window):
        is_gain = self.changes > 0
        is_loss = self.changes < 0
        gains, losses = self.changes, -1*self.changes
        gains[is_loss] = 0
        losses[is_gain] = 0
        pool = nn.AvgPool1d(window,1)
        gains = pool(gains)
        losses = pool(losses)
        rs = gains/losses
        rsi = 100 - 100/(1+rs)
        cleaned_rsi = torch.fill(torch.zeros(self.close.size(),device=device), torch.nan)
        cleaned_rsi[:, cleaned_rsi.size()[1]-rsi.size()[1]:]= rsi
        self.rsi = cleaned_rsi

    def run(self,bounds=None,tps=None,sls=None,test=True):
        if ((bounds!=None)&(tps!=None)&(sls!=None)):
            test=False
        if test:
            self.bounds = torch.fill(torch.zeros(self.close.size(),device=device), 20)
            bounds = self.bounds
            tps = torch.fill(torch.zeros(self.close.size(),device=device), 0.10)
            sls = torch.fill(torch.zeros(self.close.size(),device=device), -0.10)
        else:
            self.bounds = bounds

        self.buys = (self.rsi<bounds)
        self.buys = (self.buys & (torch.logical_not(torch.roll(self.buys,shifts=(0,-1),dims=(0,1))))).float()

        self.sells = torch.zeros(self.close.size(),device=device)
        num_stocks = self.buys.size()[0]
        period_len = self.buys.size()[1]
        trades = []
        for j in range(num_stocks):
            i = 0 
            while (i<period_len-1):
                if (self.buys[j,i]==True):
                    buy_price = self.close[j,i]
                    tp = tps[j,i]*buy_price
                    sl = sls[j,i]*buy_price
                    k = i
                    diff = 0
                    while ((diff<tp) & (diff>sl) & (k<period_len-1)):
                        k+=1
                        diff = self.close[j,k] - buy_price
                    if ((diff>tp) | (diff<sl)):
                        self.sells[j,k] = True
                        self.buys[j,i+1:k]=False
                        self.wealth[j,k] += (diff/buy_price)-0.004
                        trade_tensor = torch.ones(self.close.size(),device=device)
                        trade_tensor[j,i:k] += (self.close[j,i:k] - buy_price)/buy_price
                        trades.append(trade_tensor)
                        i = k
                    else:
                        self.buys[j,i] = False
                        i += 1
                else:
                    self.buys[j,i]==False
                    i += 1

        self.cum_wealth = torch.cumprod(self.wealth,dim=1)
        holding_durations = []
        if not trades:
            av_holding_dur = len(self.data)
        else:
            for tensor in trades:
                holding_durations.append(len(tensor[tensor!=1]))
                self.cum_wealth *= tensor
            av_holding_dur = sum(holding_durations)/len(holding_durations)
            if av_holding_dur==0:
                av_holding_dur = len(self.data)

        start_of_years = (self.years != torch.roll(self.years ,shifts=(0,1),dims=(0,1)))
        end_of_years = (self.years != torch.roll(self.years ,shifts=(0,-1),dims=(0,1)))
        year_periods = torch.logical_not(start_of_years | end_of_years)

        stock_annual_alphas = torch.zeros(num_stocks, device=device)
        win_rates = torch.zeros(num_stocks, device=device)
        average_trades = torch.zeros(num_stocks, device=device)
        max_drawdown_tensor = torch.zeros(num_stocks, device=device)

        for j in range(num_stocks):
            num_trades = len(self.wealth[j,:][(self.wealth[j,:]>1)|(self.wealth[j,:]<1)])
            start_wealth_vals = self.cum_wealth[j,:][start_of_years[j,:]]
            end_wealth_vals = self.cum_wealth[j,:][end_of_years[j,:]]
            start_close_vals = self.close[j,:][start_of_years[j,:]]
            end_close_vals = self.close[j,:][end_of_years[j,:]]
            yearly_returns = end_wealth_vals/start_wealth_vals -1
            buy_hold_yearly_returns = end_close_vals/start_close_vals -1
            yearly_alpha = yearly_returns-buy_hold_yearly_returns
            stock_annual_alphas[j] = torch.mean(yearly_alpha)
            if(num_trades==0):
                win_rates[j] = 0.0
                average_trades[j] = 0.0
            else:
                win_rate = len(self.wealth[j,:][self.wealth[j,:]>1])/num_trades
                win_rates[j] = win_rate
                average_trade = torch.mean(self.wealth[j,:][(self.wealth[j,:]>1)|(self.wealth[j,:]<1)])
                average_trades[j] = average_trade

            i=0
            max_drawdowns = []
            while (i<period_len):
                if year_periods[j,i] == False:
                    k=i+1
                    while (year_periods[j,k] != False):
                        k+=1
                    max_drawdown = (1-self.cum_wealth[j,i:k+1]/self.cum_wealth[j,i]).min().item()
                    max_drawdowns.append(max_drawdown)
                    i=k+1
            max_drawdown_tensor[j] = min(max_drawdowns)

        cum_wealth_change = (torch.roll(self.cum_wealth ,shifts=(0,1),dims=(0,1))[:,1:] - self.cum_wealth[:,1:])/self.cum_wealth[:,1:]
        benchmark_change = (torch.roll(self.close[:,:]/self.close[:,:] ,shifts=(0,1),dims=(0,1))[:,1:] - (self.close[:,:]/self.close[:,:])[:,1:])/(self.close[:,:]/self.close[:,:])[:,1:]
        if (cum_wealth_change!=0).sum() > 0:
            std = torch.std(cum_wealth_change[cum_wealth_change!=0]-benchmark_change[cum_wealth_change!=0]).item()
        else: 
            std = torch.ones(1,device=device).item()
        
        av_overall_return = torch.mean(self.cum_wealth[:,-1]/self.cum_wealth[:,0])/start_of_years.size()[1]
        av_overall_alpha = (av_overall_return-torch.mean(self.close[j,-1]/self.close[j,0]))/start_of_years.size()[1]
        ir_ratio = av_overall_alpha/std
        av_annual_alpha = torch.mean(stock_annual_alphas)
        av_average_trades = torch.mean(average_trades)
        av_winrate = torch.mean(win_rates)
        av_max_drawdown = torch.mean(max_drawdown_tensor)
        holding_dur_score = torch.tensor(3/av_holding_dur,device=device)
        x = torch.stack([av_overall_return,av_overall_alpha,av_annual_alpha,av_average_trades,av_winrate,av_max_drawdown,holding_dur_score,ir_ratio])
        fitness_score = torch.nanmean(x)
        if fitness_score<0:
            fitness_score=0
        return fitness_score**2

    def plot(self):
        codes = list(self.data.columns.get_level_values(1).unique())
        ref_data = self.data.drop(columns=['Volume'])
        y_vals = self.cum_wealth.to('cpu').numpy()
        rsi_vals = self.rsi.to('cpu').numpy()
        rsi_bounds_vals = self.bounds.to('cpu').detach().numpy()
        buy_prices = (self.buys*self.close).to('cpu').numpy()
        sell_prices = (self.sells*self.close).to('cpu').numpy()

        if len(codes)==1:
            fig,axes = plt.subplots(nrows=2, ncols=len(codes), sharex=True)    
            code = codes[0]   
            axes[0].plot(ref_data.index,y_vals[0,:], color='b')
            axes[0].set_title(code)
            ax2 = axes[0].twinx()
            ax2.plot(ref_data['Close'][code],alpha=0.2,color='k')
            open_positions_y = buy_prices[0,:][buy_prices[0,:]!=0]
            open_positions_x = ref_data.index[buy_prices[0,:]!=0]
            close_positions_y = sell_prices[0,:][sell_prices[0,:]!=0]
            close_positions_x = ref_data.index[sell_prices[0,:]!=0]
            ax2.scatter(open_positions_x,open_positions_y, s=10,color='g',alpha=0.3,marker='^')
            ax2.scatter(close_positions_x,close_positions_y, s=10,color='r',alpha=0.3,marker='v')
            ax2.set_ylabel('Stock Price (IDR)')
            axes[1].plot(ref_data.index,rsi_vals[0,:])
            axes[1].plot(ref_data.index,rsi_bounds_vals[0,:])
        elif len(codes)<=3:
            fig,axes = plt.subplots(nrows=2, ncols=len(codes), sharex=True)   
            for idx,code in enumerate(codes):     
                axes[0,idx].plot(ref_data.index,y_vals[idx,:], color='b')
                axes[0,idx].set_title(code)
                ax2 = axes[0,idx].twinx()
                ax2.plot(ref_data['Close'][code],alpha=0.2,color='k')
                open_positions_y = buy_prices[idx,:][buy_prices[idx,:]!=0]
                open_positions_x = ref_data.index[buy_prices[idx,:]!=0]
                close_positions_y = sell_prices[idx,:][sell_prices[idx,:]!=0]
                close_positions_x = ref_data.index[sell_prices[idx,:]!=0]
                ax2.scatter(open_positions_x,open_positions_y, s=10,color='g',alpha=0.3,marker='^')
                ax2.scatter(close_positions_x,close_positions_y, s=10,color='r',alpha=0.3,marker='v')
                ax2.set_ylabel('Stock Price (IDR)')
                axes[1,idx].plot(ref_data.index,rsi_vals[idx,:])
                axes[1,idx].plot(ref_data.index,rsi_bounds_vals[idx,:])
        else:
            fig,ax = plt.subplots()
            for idx,code in enumerate(codes):     
                ax.plot(ref_data.index,y_vals[idx,:], color='b',alpha=1/idx+1)

        plt.tight_layout()
        plt.show()

# bt = BackTest(data)
# bt.rsi_calc(10)
# bt.run()
# bt.plot()