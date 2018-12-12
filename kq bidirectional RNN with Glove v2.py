from Package import Newfile as nf
import pandas as pd
import numpy as np
from Package.Features import *
# from Package.Models import *
import torch
import pickle
from torch import nn
from torch.nn.modules.padding import ConstantPad1d,ReflectionPad1d
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
from numba import jit
from matplotlib import pyplot as plt
from torch.autograd import Variable
from sklearn.utils import shuffle
from time import time
# global X,Y

## loading the glove vectors
try:
    glv
except:
    glv = load_glove_model(r"D:\STATIC\glove.6B\glove.6B.300d.txt")
DEBUG = True

def log(*argv):
    if DEBUG == True:
        try:
            print(argv)
        except:
            print("Error in printing")

def get_batches(x,y,chunks = 400):
    l = len(x)
    cnt = int(np.round(int(l)/int(chunks)))
    # log(cnt)
    remain = l%chunks
    rmt = remain/cnt
    # log(rmt)
    x_list = [x[each*cnt:(each*cnt)+cnt] for each in range(chunks+int(rmt)+2)]
    y_list = [y[each*cnt:(each*cnt)+cnt] for each in range(chunks+int(rmt)+2)]
    return x_list,y_list


class encoder_rnn(nn.Module):
    def __init__(self,input_dim, hidden_dim, layer_dim, output_dim):
        super(encoder_rnn,self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, 
                          bidirectional = True)
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim*2, output_dim) ## *2 because of birectional
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        for e in range(len(x)):
            X = torch.from_numpy(x[e])[None,None,:].float()
            if e == 0:
                # h0 = (torch.randn(self.layer_dim*2, 1, self.hidden_dim),torch.randn(self.layer_dim*2, 1, self.hidden_dim))
                h0 = torch.randn(self.layer_dim*2, 1, self.hidden_dim)
                output,hidden = self.rnn(X,h0)
            else:
                output,hidden = self.rnn(X,hidden)
        fc_output = self.fc(output)
        return self.sigmoid(fc_output)

@jit
def numba_mean(loss_list):
    return np.array(loss_list).mean()
    

def feed(x,y,backprop = False,epoch=1):
    t1 = time()
    loss_list = []
    for i,each in enumerate(x):
        X = each
        Y = torch.from_numpy(np.array(y[i])).float().to(device)
        pred = model.forward(X)
        loss =cross(pred,Y.float()).to(device)
        if backprop == True:
            loss.backward()
            optimizer.step()
        loss_list.append(loss.item())
    print("T:",str(time()-t1))
    return loss_list


#%%
## Device agnostic code            
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

try:glv
except: glv = load_glove_model(r"D:\STATIC\glove.6B\glove.6B.300d.txt")

## loading the training data
q = nf("train_sample_20_r42.csv",ascii = True)
q.data = shuffle(q.data).reset_index(drop = True)
log("Total rows ",q.data.shape)
# q.data = q.data[:200000]
log(q.data.groupby("target")["qid"].count())
log("Considered rows ",q.data.shape)
q.content_col = "question_text"
q.preprocess(remove_stopwords = False,lemmatize = False)
log("preprocessing words complete")
q.data["x"]  = q.data["p_content"]
q.data["y"] = q.data["target"]

## test train split
x,xt,y,yt = train_test_split(q.data['x'],q.data['y'],test_size=0.005, random_state=42)



x = x.reset_index(drop = True)
xt = xt.reset_index(drop = True)
y = y.reset_index(drop = True)
yt =yt.reset_index(drop = True)

## readying the vectors for the test data
xt = [get_vectors(each,glv,pad= None) for each in xt]

log("Vectorization complete")

## obtaining batches
x_batches,y_batches = get_batches(x,y)
x_batches = [each.reset_index(drop = True) for each in x_batches if len(each)!=0]
y_batches = [each.reset_index(drop = True) for each in y_batches if len(each)!=0]

log("splitting the batches is complete")  
log("The shape of each batch is ",x_batches[0].shape)      
log("The shape of test data is ",len(xt))      

## initiate a model   
model  = encoder_rnn(300,30,3,1)
model = model.to(device)
epoch = 100
lr = 0.00001
checkpoint = 3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer.zero_grad()
cross = nn.BCELoss()

log("Model training starts")
    
training_cost_list = []
testing_cost_list = []
epoch_list = []
batch_list = []
for each in range(epoch):
    for j,e in enumerate(x_batches):
        x, y = x_batches[j],y_batches[j]
        x = [get_vectors(each,glv,pad=None) for each in x]
        train_cost = numba_mean(feed(x,y,backprop = True,epoch = each))  
        test_cost = numba_mean(feed(xt,yt,backprop = False,epoch = each))
        
        log("Epoch num - ",each," and batch num ",j)
        log(train_cost)
        log(test_cost)        
        log("================================================")
        
        
        training_cost_list.append(train_cost)
        testing_cost_list.append(test_cost)
        epoch_list.append(each)
        batch_list.append(j)
        
        ## logging
        if j %checkpoint == 0:
            logs = pd.DataFrame()
            logs["training_cost"] = training_cost_list
            logs["testing_cost"] = testing_cost_list
            logs["epoch_list"] = epoch_list
            logs["batch_list"] = batch_list
            try:
                logs.to_csv("log.csv")
            except:
                pass
        
        save(model.state_dict,r"D:\Quora insincere\models\BRNN_g300d_proper_v1_"+str(each)+"_"+str(j))
        
    