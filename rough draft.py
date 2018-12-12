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
from sklearn.utils import shuffle


sin = nf("sin.csv")
insin = nf("insin.csv")

sample = nf(data = sin.data.sample(frac = 0.20,random_state = 42))

print(sample.shape)

sample2 = sample.append(object = insin)
print(sample2.shape)

sample2.csv("train_sample_20_r42.csv")

#%%