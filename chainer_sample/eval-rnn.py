#!/usr/bin/env python

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
                        optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import math
import sys
argvs = sys.argv

vocab = {}

def load_data(filename):
    global vocab
    words = open(filename).read().replace('\n', '<eos>').strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    return dataset

class MyRNN(chainer.Chain):
    def __init__(self, v, k):
        super(MyRNN, self).__init__(
            embed = L.EmbedID(v, k),
            H  = L.Linear(k, k),
            W = L.Linear(k, v),
        )
    def __call__(self, s):
        accum_loss = None
        v, k = self.embed.W.data.shape
        h = Variable(np.zeros((1,k), dtype=np.float32))
        for i in range(len(s)):
            next_w_id = eos_id if (i == len(s) - 1) else s[i+1]
            tx = Variable(np.array([next_w_id], dtype=np.int32))
            x_k = self.embed(Variable(np.array([s[i]], dtype=np.int32)))
            h = F.tanh(x_k + self.H(h))
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

train_data = load_data('ptb.train.txt')

demb = 100
def cal_ps(model, s):
    h = Variable(np.zeros((1,demb), dtype=np.float32))
    sum = 0.0
    for i in range(1,len(s)):
        w1, w2 = s[i-1], s[i]
        x_k = model.embed(Variable(np.array([w1], dtype=np.int32)))
        h = F.tanh(x_k + model.H(h))
        yv = F.softmax(model.W(h))
        pi = yv.data[0][w2]
        sum -= math.log(pi, 2)
    return sum
        
eos_id = vocab['<eos>']
max_id = len(vocab)
test_data = load_data('ptb.test.txt')
test_data = test_data[0:1000]

model = MyRNN(len(vocab), demb)
serializers.load_npz(argvs[1], model)
sum = 0.0
wnum = 0
s = []
unk_word = 0    
for pos in range(len(test_data)):
    id = test_data[pos]
    s.append(id)
    if (id > max_id):
        unk_word = 1
    if (id == eos_id):
        if (unk_word != 1):        
            ps = cal_ps(model, s)
            sum += ps
            wnum += len(s) - 1
        else:
            unk_word = 0                
        s = []    
print math.pow(2, sum / wnum)




