from Package import Newfile as nf
import pandas as pd
import numpy as np
from Package.Features import *
from Package.Models import *
import torch
import pickle
from torch import nn
from torch.nn.modules.padding import ConstantPad1d,ReflectionPad1d
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
from numba import jit
from matplotlib import pyplot as plt
from torch.autograd import Variable
glv = load_glove_model(r"D:/SENTIMENT CLASSIFIER/glove.6B.50d.txt")
DEBUG = True
def log(*argv):
    if DEBUG == True:
        try:
            print(argv)
        except:
            print("Error in printing")
        


#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

q = nf("train.csv",ascii = True)
q.data = q.data[:30]
q.content_col = "question_text"
q.preprocess(remove_stopwords = False,lemmatize = False)
v = [get_vectors(each,glv,pad=None) for each in q.data["p_content"]]
q.data["x"] = v
q.data["y"] = q.data["target"]
# q.csv("processed_train.csv")

# q = nf("processed_train.csv")
# q.data = q.data[:80]

# q.data[:50].to_csv("sample_train.csv",index = None)

x,xt,y,yt = train_test_split(q.data['x'],q.data['y'],test_size=0.2, random_state=42)

x = x.reset_index(drop = True)
xt = xt.reset_index(drop = True)
y = y.reset_index(drop = True)
yt =yt.reset_index(drop = True)



class encoder_rnn(nn.Module):
    def __init__(self,input_dim, hidden_dim, layer_dim, output_dim):
        super(encoder_rnn,self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, 
                          bidirectional = True)
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim*2, output_dim) ## *2 because of birectional
        self.sig = nn.Sigmoid()
        
    def forward(self,x):
        self.h0 =Variable(torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim))
            # log("expect")
          ##*2 because of bidirectional
        output,hidden = self.rnn(x,self.h0)
        # self.h0 = hidden
        op = self.fc(output)
        last_op = op[0][op.shape[1]-1]
        return self.sig(last_op)
    
    # def forward(self,x,hidden):
    #     output,hidden = self.rnn(x,hidden)
    #     op = self.fc(output)
    #     last_op = op[0][op.shape[1]-1]
    #     return self.sig(last_op),hidden
    
model  = encoder_rnn(50,20,3,1)
model = model.to(device)
epoch = 100
lr = 0.00001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
optimizer.zero_grad()
cross = nn.BCELoss()

@jit
def numba_mean(loss_list):
    return np.array(loss_list).mean()
    
    plt.figure()
# plt.show()
def feed(x,y,backprop = False,epoch=1):
    loss_list = []
    for i,each in enumerate(x):
        X = torch.from_numpy(each)[None,:,:].float().to(device)
        Y = torch.from_numpy(np.array(y[i])).float().to(device)
        # log(X.shape)
        pred = model.forward(X).to(device)
        loss =cross(pred,Y.float()).to(device)
        if backprop == True:
            if epoch==0:
                loss.backward(retain_graph = True)
            else:
                loss.backward()
            optimizer.step()
        # del model.h0
        loss_list.append(loss.item())
    return loss_list


for each in range(epoch):
    
    train_cost = numba_mean(feed(x,y,backprop = True,epoch = each))
    log(each)
    log(train_cost)
    
    test_cost = numba_mean(feed(xt,yt,backprop = False,epoch = each))
    log(test_cost)
    # plt.scatter(x = each,y = train_cost,color = "blue")
    # plt.scatter(x = each,y = test_cost,color = "red",)
    
    log("================================================")
    