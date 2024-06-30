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
    inputs = pd.concat([inputs,inputs.shift(1),inputs.shift(2)],axis=1)
    return inputs

class NN(nn.Module):
    def __init__(self,num_h):
        super(NN,self).__init__()
        self.hidden_layers = num_h
        self.fcstart = nn.Linear(21,10)
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

def plot_best_model(best_model,data,inputs,codes):
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
    fitness_score = bt.run(bounds,tps,sls)
    print(f"Best Fitness Score: {fitness_score}")
    bt.plot()
    

def main(data,inputs,codes,population_num=8,generations=20,parent_num=2):
    try:
        population = [NN(3).to(device) for i in range(population_num)]
        for model in population:
            model.load_state_dict(torch.load('model.pth'))
            model.eval()
    except:
        population = [NN(3).to(device).eval() for i in range(population_num)]
    if population_num<=2:
        raise Exception("Need Higher Population Num")

    best_model = {"Model":NN(3).to(device).eval(),"Fit":0}

    with torch.no_grad():
        for i in range(1,generations+1):
            fitnesses = calc_fitness(population,data,inputs,codes)
            max_fit = max(fitnesses)
            if max_fit>best_model['Fit']:
                idx_max = fitnesses.index(max_fit)
                best_model['Model'].load_state_dict(population[idx_max].state_dict())
                best_model['Fit'] = max_fit
            print(f"Gen: {i}/{generations}, Max Fitness: {max(fitnesses)}")
            parents = select_parents(population,fitnesses,parent_num)
            population = crossover(parents,population)
            mutate(population)
        torch.save(best_model['Model'].state_dict(),'model.pth')
        plot_best_model(best_model['Model'],data,inputs,codes)

if __name__ == "__main__":
    turnovers = (data['Close']*data['Volume'])
    turnovers = turnovers.loc[:,(1-(turnovers.rolling(10).mean()==0).any(axis=0)).astype(bool)].mean()
    codes = list(turnovers[turnovers>10**9].index)
    data = data.loc[:,data.columns.get_level_values(1).isin(codes[:2])].dropna()
    inputs = gen_inputs(data)
    codes = codes[:2]
    main(data,inputs,codes,generations=5)
    # best_model = NN(3).to(device)
    # load = torch.load('model.pth')
    # best_model.load_state_dict(load)
    # best_model.eval()
    # plot_best_model(best_model,data,inputs,codes)


