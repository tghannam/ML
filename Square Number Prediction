import pandas as pd
import numpy as np
import gmpy2 as gm
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, auc, confusion_matrix, r2_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tqdm import trange
import pydotplus
from IPython.display import Image
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn import neighbors
import matplotlib.patches as mpatches
import pygraphviz as graphviz
from sklearn.tree import export_graphviz
import matplotlib.patches as mpatches
import math
import torch as tc
import torch.nn as nn
#import wandb
import os
from xgboost import XGBRegressor
import string
import random
from sklearn.model_selection import StratifiedKFold


def digit_root(n): 
    return (n - 1) % 9 + 1

def perfect_square(x):
    if gm.is_square(x):
        return np.sqrt(x)
    
def magnitude(x):
    if x != 0 :
        return int(math.floor(math.log10(x)))+1
    else:
        return 1

def droot_ps(x):
    ''' A function that takes a number and returns its previous PSN (X_minus), next PSN (X_plus) along with the digital root of x D(x), ordr of magnitude of x, digtial roots of the PSN (D_P, D_N)
    the ditance form X to the PSN (diff_plus, diff_minus) and their digital roots (Ddp Ddn)'''
    x_plus=[]
    x_minus = []
    for i in np.arange(x):
        x_p = x+i
        x_m = x-i
        if gm.is_square(int(x_p)):
            x_plus.append(x_p)
        if gm.is_square(int(x_m)):
            x_minus.append(x_m)
        if (len(x_plus) >= 1 & len(x_minus) >=1):
            break
    
    diff_plus = x_plus[0]-x
    diff_minus = x-x_minus[0]
    D_p = digit_root(x_plus[0])
    D_m = digit_root(x_minus[0])
    D_x = digit_root(x)
    Ddp = digit_root(diff_plus)
    Ddn = digit_root(diff_minus)
    Order = magnitude(x)
    return (x, D_x, Order, x_plus, D_p, diff_plus, Ddp, x_minus, D_m, diff_minus, Ddn)

# Shallow Learning

def id_generator(size=12, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in np.arange(size))

HASH_NAME = id_generator(size=12)


CONFIG = {"seed": 42,
          "model_name": "square_predict_shallow",
          "learning_rate":0.15,
          "hash_name": HASH_NAME,
          'eval_metric':'rmsle',
          'max_depth':5,
          'n_estimators':6000,
          'n_fold':2
          }


CONFIG['group'] = f'{HASH_NAME}-Baseline'

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(CONFIG['seed'])

df= pd.read_csv('C:\\Users\\gtala\\Desktop\\Theon\\sqaure_pred_100K-10000K.csv', index_col=0) 
df = df.sample(frac=1)

cat_cols = [  ] #'D_N', 'D_P', 'D_d_N', 'D_d_P', 'D_x', 
cont_cols = ['dP+dN'] #'O','ratio' 'dP-dN', 'd_P', 'd_N'
X = df[cat_cols+cont_cols]
y_col = 'P'
y = df[y_col]

skf = StratifiedKFold(n_splits=CONFIG['n_fold'], shuffle=True, random_state=CONFIG['seed'])

for fold, ( _, val_) in enumerate(skf.split(X=df, y=y)):
    df.loc[val_ , "kfold"] = int(fold)

df["kfold"] = df["kfold"].astype(int)

for fold in np.arange(0, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_test = df[df.kfold == fold].reset_index(drop=True)

X_train = df_train[cat_cols+cont_cols]
y_train = df_train[y_col]
X_test = df_test[cat_cols+cont_cols]
y_test = df_test[y_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=True, test_size = 0.5)
XGB = XGBRegressor(eval_metric='rmse',tree_method='gpu_hist', n_estimators=CONFIG['n_estimators'],  max_depth= CONFIG['max_depth'], learning_rate=CONFIG["learning_rate"] )
XGB.fit(X_train, y_train)

SqrtMSE = np.sqrt(mean_squared_error(y_test, XGB.predict(X_test)))
SqrtMSE/(df['x'].max()-df['x'].min())*100

y_eval = eval_.values.ravel()
trace_1 = go.Scatter(x = y_test.index.values[:5000], y =y_test.values.ravel()[:5000], name='Exact label')
trace_2 = go.Scatter(x= y_test.index.values[:5000], y = y_eval[:5000], name='Predicted label')
data = [trace_1, trace_2]
layout = go.Layout(title='Exact vs. Predicted Labels')
fig = go.Figure(data)
fig.layout = layout
fig.show()

y_pred = pd.Series(XGB.predict(X_test))
data = [go.Scatter(x=y_test.index.values[:100000], y =y_test- y_pred)]
fig = go.Figure(data)
fig.show()

# Save and deploy model
XGB.save_model("model.txt")
model2 = XGBRegressor()
model2.load_model("model.txt")
x = droot_ps_test(int(input()))
x_test = pd.DataFrame.from_dict(x)
print(x, x_test['N'], int(pd.Series(model2.predict(x_test[['d_P', 'd_N']]))))

x = droot_ps_test(int(input()))
x_test = pd.DataFrame.from_dict(x)
print(x, x_test['N'], int(pd.Series(model2.predict(x_test[['d_P', 'd_N']]))))


# Deep Learning

import torch as tc
import string
import random
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR, ExponentialLR
import gc
# Utils
from tqdm import tqdm
from collections import defaultdict
import time
import copy
# Sklearn Imports
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold

from colorama import Fore, Back, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

df= pd.read_csv('C:\\Users\\gtala\\Desktop\\Theon\\sqaure_pred_100K-10000K.csv', index_col = None)

'''cat_cols = ['D_x', 'D_d_P', 'D_P']
cont_cols = ['d_P', 'O']'''

#cat_cols = ['D_x', 'D_d_P', 'D_P', 'D_d_N', 'D_N']
cont_cols = ['d_P', ] #'O', 'd_N'

y_col = ['P']

#cats = np.stack([df[col].cat.codes.values for col in cat_cols], 1)
cont = np.stack([df[col].values for col in cont_cols], 1)
cont = tc.tensor(cont, dtype=tc.float32).cuda()
# cats = tc.tensor(cats, dtype=tc.int64).cuda()
y=tc.tensor(df[y_col].values, dtype=tc.float).reshape(-1, 1).cuda()
# cats.shape, cont.shape, y.shape

batch_size = len(df)
test_size = int(batch_size * .3)

# cat_train = cats[:batch_size-test_size]
# cat_test = cats[batch_size-test_size:batch_size]
con_train = cont[:batch_size-test_size]
con_test = cont[batch_size-test_size:batch_size]
y_train = y[:batch_size-test_size]
y_test = y[batch_size-test_size:batch_size]
# len(cat_train), len(cat_test)

def id_generator(size=12, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in np.arange(size))

HASH_NAME = id_generator(size=12)
print(HASH_NAME)

CONFIG = {"seed": 2022,
          "epochs": 500,
          "model_name": "square_predict",
          "train_batch_size": 32,
          "valid_batch_size": 64,
          "learning_rate":0.05,
          "scheduler": 'CosineAnnealingLR',
          "min_lr": 1e-6,
          "T_max": 500,
          "weight_decay": 1e-6,
          "n_fold": 2,
          "n_accumulate": 1,
          "num_classes": 1,
          "margin": 0.5,
          "device": tc.device("cuda:0" if tc.cuda.is_available() else "cpu"),
          "hash_name": HASH_NAME,
          "layers": [500, 500]
          }


CONFIG['group'] = f'{HASH_NAME}-Baseline'

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    tc.manual_seed(seed)
    tc.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    tc.backends.cudnn.deterministic = True
    ''' This is set to make sure that we get the same exact result for every similar run '''
    tc.backends.cudnn.benchmark = True 
    '''It enables benchmark mode in cudnn.benchmark mode is good whenever input sizes for the network do not vary. This way, cudnn will look for the optimal set of algorithms 
        for that particular configuration (which takes some time). This usually leads to faster runtime. But if input sizes changes at each iteration, 
        then cudnn will benchmark every time a new size appears, possibly leading to worse runtime performances.'''
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(CONFIG['seed'])

class PSN(nn.Module):
    def __init__(self, n_cont, out_sz, layers, p=0.4,): #emb_szs, 
        super().__init__()
        # self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
        # self.emb_drop = nn.Dropout(p)
        #self.bn_cont = nn.BatchNorm1d(n_cont)
        
        layerlist = []
        #n_emb = sum((nf for ni,nf in emb_szs))
        n_in =  n_cont #n_emb +
        
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.ReLU())
            #layerlist.append(nn.Linear(i,i)) 
            #layerlist.append(nn.ReLU())
            #layerlist.append(nn.BatchNorm1d(i))
            #layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],out_sz))
            
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self,  x_cont): #x_cat,
        embeddings = []
        # for i,e in enumerate(self.embeds):
        #     embeddings.append(e(x_cat[:,i]))
        # x = tc.cat(embeddings, 1)
        # #x = self.emb_drop(x)
        
        # #x_cont = self.bn_cont(x_cont)
        # x = tc.cat([x, x_cont], 1)
        x = x_cont
        x = self.layers(x)
        return x

tc.manual_seed(20)
model = PSN( cont.shape[1], 1, CONFIG['layers'], p=0.4).cuda() #emb_szs,

criterion = nn.MSELoss()
optimizer = tc.optim.Adam(model.parameters(), lr = CONFIG['learning_rate'])

import time
start_time = time.time()

epochs = CONFIG['epochs']
losses = []

for i in np.arange(epochs):
    model.train()
    #wandb.watch(model)
    i+=1
    y_pred = model(con_train) #cat_train, 
    loss = tc.sqrt(criterion(y_pred, y_train)) # RMSE
    losses.append(loss.item())
    #wandb.log({"loss": loss})
    # a neat trick to save screen space:
    if i%25 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    
    optimizer.step()
    #scheduler.step()

print(f'epoch: {i:3}  loss: {loss.item():10.8f}') # print the last line
print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed

model.eval()
with tc.inference_mode():
    eval_loss = []
    y_val = model(con_test) #cat_test, 
    loss = tc.sqrt(criterion(y_val, y_test))
    eval_loss.append(loss.item())
print(f'RMSE: {loss:.8f}')

print(f'{"PREDICTED":>12} {"ACTUAL":>8} {"DIFF":>8}')
for i in np.arange(1, 50):
    diff = np.abs(y_val[i].item()-y_test[i].item())
    print(f'{i+1:2}. {y_val[i].item():8.4f} {y_test[i].item():8.4f} {diff:8.4f}')
