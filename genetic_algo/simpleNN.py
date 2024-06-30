import torch
import torch.nn as nn
from bt import BackTest
import pandas as pd
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

data = pd.read_pickle('data.pkl')
data = data[(data.index > '2015-01-01') & (data.index < '2023-12-31')]
codes = data.columns.get_level_values(1).unique()

def gen_inputs(data):
    i1 = ((data['Close'] - data['Open'])/data['Open']).shift(1)
    i1.name = 'O-C Change'
    i2 = ((data['Close'] - data['Close'].shift(1))/data['Close'].shift(1)).shift(1)
    i2.name = 'C-C Change'
    i3 = (data['Close']+data['Open'])*data['Volume']/2
    i3 = (i3/i3.rolling(5).mean()).shift(1)
    i3.name = '5 Day Rel Turnover'
    i4 = ((data['Low']-data['Low'].shift(1))/data['Low'].shift(1)).shift(1)
    i4.name = 'L-L Change'
    i5 = ((data['High']-data['High'].shift(1))/data['High'].shift(1)).shift(1)
    i5.name = 'H-H Change'
    i6 = ((data['High']-data['Close'])/data['Close']).shift(1)
    i6.name = 'H-C Change'
    i7 = ((data['Close']-data['Low'])/data['Close']).shift(1)
    i7.name = 'L-C Change'

    inputs = pd.concat([i1,i2,i3,i4,i5,i6,i7],axis=1)
    inputs = (inputs-inputs.mean())/inputs.std()
    return inputs

class NN(nn.Module):
    def __init__(self,num_h):
        super(NN,self).__init__()
        self.hidden_layers = num_h
        self.fcstart = nn.Linear(7,10)
        self.fcmid = nn.Linear(10,10)
        self.fcend = nn.Linear(10,3)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.fcstart(x)
        for i in range(self.hidden_layers):
            x = self.fcmid(x)
        x = self.sigmoid(self.fcend(x)/3)
        return x

def calc_fitness(population,data,inputs,codes):
    fitnesses = []
    bt=BackTest(data)
    bt.rsi_calc(14)
    for NeuralNet in population:
        bounds = torch.zeros(data['Close'].transpose().shape,device=device,dtype=torch.float32)
        tps = torch.zeros(data['Close'].transpose().shape,device=device,dtype=torch.float32)
        sls = torch.zeros(data['Close'].transpose().shape,device=device,dtype=torch.float32)
        for idx,code in enumerate(codes):
            input_tensor = torch.tensor(inputs[code].values,dtype=torch.float32,device=device)
            out = torch.transpose(NeuralNet(input_tensor),0,1)
            bounds[idx,:] = out[0,:]*40
            tps[idx,:] = out[1,:]/3
            sls[idx,:] = -out[2,:]/3
        fitness_score = bt.run(bounds=bounds,tps=tps,sls=sls).item()
        fitnesses.append(fitness_score)
        bt.reset()
        torch.cuda.empty_cache()
    return fitnesses

def select_parents(population,fitnesses,parent_num=4):
    parents = random.choices(population,fitnesses,k=parent_num)
    return parents

def crossover(parents, population):
    for i in range(len(population)):
        child = population[i]
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child_state_dict = child.state_dict()
        parent1_state_dict = parent1.state_dict()
        parent2_state_dict = parent2.state_dict()
        rand = random.uniform(0,1)
        for key in parent1_state_dict:
            child_state_dict[key] = parent1_state_dict[key]*rand + parent2_state_dict[key]*(1-rand)
        child.load_state_dict(child_state_dict)
        population[i] = child
    return population

def mutate(population,mutation_rate=0.5):
    for i in range(len(population)):
        if random.uniform(0,1)<mutation_rate:
            for param in population[i].parameters():
                rand = torch.randn_like(param)
                rand = rand/torch.abs(rand).max()*2
                param.data.add_(rand*param.min())

def plot_best_model(population,data,inputs,codes):
    fitnesses = calc_fitness(population,data,inputs,codes)
    max_fit = max(fitnesses)
    idx_max = fitnesses.index(max_fit)
    best_model = population[idx_max]
    bt=BackTest(data)
    bt.rsi_calc(14)
    bounds = torch.zeros(data['Close'].transpose().shape,device=device,dtype=torch.float32)
    tps = torch.zeros(data['Close'].transpose().shape,device=device,dtype=torch.float32)
    sls = torch.zeros(data['Close'].transpose().shape,device=device,dtype=torch.float32)
    for idx,code in enumerate(codes):
        input_tensor = torch.tensor(inputs[code].values,dtype=torch.float32,device=device)
        out = torch.transpose(best_model(input_tensor),0,1)
        bounds[idx,:] = out[0,:]*40
        tps[idx,:] = out[1,:]/3
        sls[idx,:] = -out[2,:]/3
    bt.run(bounds,tps,sls)
    bt.plot()

def main(data,inputs,codes,population_num=8,generations=20,parent_num=2):
    population = [NN(3).to(device).eval() for i in range(population_num)]
    if population_num<=2:
        raise Exception("Need Higher Population Num")

    with torch.no_grad():
        for i in range(1,generations+1):
            fitnesses = calc_fitness(population,data,inputs,codes)
            print(f"Gen: {i}/{generations}, Max Fitness: {max(fitnesses)}")
            parents = select_parents(population,fitnesses,parent_num)
            population = crossover(parents,population)
            mutate(population)
        plot_best_model(population,data,inputs,codes)

if __name__ == "__main__":
    data = data.loc[:,data.columns.get_level_values(1).isin(codes[:1])].dropna()
    inputs = gen_inputs(data)
    codes = codes[:1]
    main(data,inputs,codes,generations=10)

