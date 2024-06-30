from backtesting import Backtest, Strategy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
'''
Function to get stock price data of relevant indonesian stocks
'''

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def data_download():
    datas = []
    df = pd.read_pickle('data.pkl')
    stock_codes = df.columns.get_level_values(1)
    for code in stock_codes:
        data = df.loc[:,df.columns.get_level_values(1) == code]
        data.columns = data.columns.get_level_values(0)
        datas.append(data.dropna())
    return datas

class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(13,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,30)
        self.fc4 = nn.Linear(30,5)
        self.relu = nn.ReLU()
        self.sigmoid  = nn.Sigmoid()
    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(self.fc4(x))
        return x

def smadiff(close:np.array, n_sma:int): 
    return_array = pd.Series(close).rolling(n_sma).mean().to_numpy()
    return_array = ((return_array-close)/close)*100
    return return_array

def rsi(open:np.array, close:np.array, n_rsi:int) -> np.array:
    diff = close-open
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

def sma5_bounds(close:np.array, MLModelOutput) -> np.array:
    len1 = len(close)
    len2 = MLModelOutput.size()[0]
    len_diff = len1-len2
    bounds = MLModelOutput[:,0]*20
    nans = torch.zeros(len_diff, device=device)
    nans[:] = torch.nan
    return_array = torch.concatenate([nans,bounds]).cpu().detach().numpy()
    return return_array

def rsi14_bounds(close:np.array, MLModelOutput) -> np.array:
    len1 = len(close)
    len2 = MLModelOutput.size()[0]
    len_diff = len1-len2
    bounds = MLModelOutput[:,1]*40
    nans = torch.zeros(len_diff, device=device)
    nans[:] = torch.nan
    return_array = torch.concatenate([nans,bounds]).cpu().detach().numpy()
    return return_array

def rsi30_bounds(close:np.array, MLModelOutput) -> np.array:
    len1 = len(close)
    len2 = MLModelOutput.size()[0]
    len_diff = len1-len2
    bounds = MLModelOutput[:,2]*40
    nans = torch.zeros(len_diff, device=device)
    nans[:] = torch.nan
    return_array = torch.concatenate([nans,bounds]).cpu().detach().numpy()
    return return_array

def tps(close:np.array, MLModelOutput) -> np.array:
    len1 = len(close)
    len2 = MLModelOutput.size()[0]
    len_diff = len1-len2
    tps = (4.02 - MLModelOutput[:,3])/3
    nans = torch.zeros(len_diff, device=device)
    nans[:] = torch.nan
    return_array = torch.concatenate([nans,tps]).cpu().detach().numpy()
    return return_array

def sls(close:np.array, MLModelOutput) -> np.array:
    len1 = len(close)
    len2 = MLModelOutput.size()[0]
    len_diff = len1-len2
    sls = (2-MLModelOutput[:,4])/2
    nans = torch.zeros(len_diff, device=device)
    nans[:] = torch.nan
    return_array = torch.concatenate([nans,sls]).cpu().detach().numpy()
    return return_array

#Each row will be 13 data points
def initial_data(data, MLModel):
    close = pd.Series(np.array(data.Close))
    open = pd.Series(np.array(data.Open))
    high = pd.Series(np.array(data.High))
    low = pd.Series(np.array(data.Low))
    vol = pd.Series(np.array(data.Volume))
    daily_turnover = vol*open
    
    daily_turnover_change = (daily_turnover-daily_turnover.shift(1))/daily_turnover.shift(1)
    prior_daily_returns = ((close-open)/open * 100).shift(1)
    prior_7_day_volatility = prior_daily_returns.rolling(7).std()
    prior_close_to_close = ((close-close.shift(1))/close.shift(1) * 100).shift(1)
    prior_weekly_returns = ((close-close.shift(7))/close * 100).shift(1)
    prior_daily_high_low = ((high-low)/low * 100).shift(1)
    prior_weekly_high_low = ((high.rolling(7).max()-low.rolling(7).min())/low.rolling(7).min() * 100).shift(1)
    prior_high_high_change = ((high-high.shift(1))/high.shift(1)*100).shift(1)
    prior_low_low_change = ((low-low.shift(1))/low.shift(1)*100).shift(1)
    prior_weekly_high_high = ((high-high.shift(7))/high.shift(7)*100).shift(1)
    prior_weekly_low_low = ((low-low.shift(7))/low.shift(7)*100).shift(1)
    prior_close_low = ((close-low)/low*100).shift(1)
    prior_close_high = ((high-close)/close*100).shift(1)

    list_of_vals = [daily_turnover_change,
                    prior_daily_returns,
                    prior_7_day_volatility,
                    prior_close_to_close,
                    prior_weekly_returns,
                    prior_daily_high_low,
                    prior_weekly_high_low,
                    prior_high_high_change, 
                    prior_low_low_change,
                    prior_weekly_high_high,
                    prior_weekly_low_low,
                    prior_close_low,
                    prior_close_high]
    return_array = pd.concat(list_of_vals, axis=1).dropna()
    return_array = torch.tensor(return_array.values, dtype=torch.float32, device=device)
    
    return np.array(close), np.array(open), MLModel(return_array)
    
class RSIOverBoughtOverSold(Strategy):
    def init(self, MLModel):
        close, open, output = initial_data(self.data, MLModel)

        self.rsi14_bounds_vals = self.I(rsi14_bounds,close, output)
        self.rsi30_bounds_vals = self.I(rsi30_bounds,close, output)
        self.sma5_bounds_vals = self.I(sma5_bounds, close, output)
        self.tp = self.I(tps, close,output)
        self.sl = self.I(sls, close,output)
        self.rsi14 = self.I(rsi, open,close,14)
        self.rsi30 = self.I(rsi, open,close,30)
        self.sma5_diff = self.I(smadiff, close, 5)

    def next(self):
        price = self.data.Close[-1]
        if ((self.rsi14<self.rsi14_bounds_vals) & 
            (self.rsi30<self.rsi30_bounds_vals) & 
            (self.sma5_diff>self.sma5_bounds_vals)):
            self.buy(tp=price*self.tp, sl=price*self.sl)

def custom_output_func(datas, MLModel:NN):
    def obtain_average_output_vals(datas=datas):
        outputs = []
        for data in datas[:1]:
            bt = Backtest(data, RSIOverBoughtOverSold(), cash=100000000, commission=0.004, exclusive_orders=True)
            output = bt.run()
            outputs.append(output)


        mean_vals = np.mean(np.stack([np.array([output.loc['Return [%]']/100,
                                                output.loc['Sharpe Ratio'],
                                                output.loc['Sortino Ratio'],
                                                output.loc['Calmar Ratio'],
                                                (50-output.loc['Win Rate [%]'])/5,
                                                output.loc['Profit Factor'],
                                                output.loc['Expectancy [%]'],
                                                output.loc['Avg. Trade [%]'],
                                                output.loc['SQN'],
                                                (50+output.loc['Max. Drawdown [%]'])/10,
                                                (output.loc['# Trades']-30)/3]) for output in outputs]), axis=0)
        if np.isnan(mean_vals).any():
            mean_vals = np.zeros(11)
            mean_vals[:] = random.uniform(0, 100)

        return mean_vals

    mean_vals = obtain_average_output_vals()
    return torch.tensor(mean_vals,device=device,dtype=torch.float32, requires_grad=True)

def custom_test(datas, MLModel:NN):
    def test(datas=datas):
        for data in datas[:1]:
            bt = Backtest(data, RSIOverBoughtOverSold, cash=100000000, commission=0.004, exclusive_orders=True)
            output = bt.run()
            bt.plot()
            print(output)
    test()

def custom_fitness_func(mean_vals:np.array):
    fitness = 0.00001/torch.exp(-1*(mean_vals-4.5)).mean()
    if fitness == float('inf'):
        fitness = torch.tensor(10000)
    return fitness

def generate_child_models(parent_models,child_models,num_models):
    for i in range(num_models):
        child = child_models[i]
        parent1 = random.choice(parent_models)
        parent2 = random.choice(parent_models)
        parent1_params = parent1.state_dict()
        parent2_params = parent2.state_dict()
        child_params = child.state_dict()
        for key in child_params.keys():
            num = random.uniform(0,1)
            child_params[key] = parent1_params[key]*(num) + parent2_params[key]*(1-num)
        child.load_state_dict(child_params)
        for param in child.parameters():
            if random.random()<0.10:
                param.data += torch.randn_like(param)*0.01
        child_models[i] = child
    return child_models


def main():
    datas = data_download()
    num_models = 20

    models = [NN().to(device) for i in range(num_models)]
    parent_models = []
    fitnesses = np.zeros(num_models)
    indexes = np.arange(num_models)

    num_epochs = 100
    for epoch in range(num_epochs):
        for i in range(num_models):
            MLModel = models[i]
            mean_vals = custom_output_func(datas,MLModel)
            fitness = custom_fitness_func(mean_vals)
            fitnesses[i] = fitness.item()
        print(f'Epoch {epoch}, Fitness: {fitnesses.max()}')
        fitnesses = fitnesses/fitnesses.sum()
        choices = np.random.choice(indexes, 5 , p = fitnesses)
        parent_models = [models[choices[i]] for i in range(len(choices))]
        models = generate_child_models(parent_models,models, num_models)

    max_idx = np.argmax(fitnesses)
    MLModel = models[max_idx]
    custom_test(datas,MLModel)

main()