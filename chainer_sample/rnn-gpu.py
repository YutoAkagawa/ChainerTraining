#!/usr/bin/env python

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
                        optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

xp = cuda.cupy  ## added

# Set data

vocab = {}

def load_data(filename):
    global vocab
    words = open(filename).read().replace('\n', '<eos>').strip().split()
    dataset = xp.ndarray((len(words),), dtype=xp.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    return dataset

train_data = load_data('ptb.train.txt')
eos_id = vocab['<eos>']

# Define model

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
        h = Variable(xp.zeros((1,k), dtype=xp.float32))
        for i in range(len(s)):
            next_w_id = eos_id if (i == len(s) - 1) else s[i+1]
            tx = Variable(xp.array([next_w_id], dtype=xp.int32))
            x_k = self.embed(Variable(xp.array([s[i]], dtype=xp.int32)))
            h = F.tanh(x_k + self.H(h))
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

# Initialize model    

demb = 100
model = MyRNN(len(vocab), demb)
cuda.get_device(0).use()       ## added 
model.to_gpu()                 ## added 
optimizer = optimizers.Adam()
optimizer.setup(model)

# Learn and Save

for epoch in range(5):
    s = []    
    for pos in range(len(train_data)):
        id = train_data[pos]
        s.append(id)        
        if (id == eos_id):
            loss = model(s)
            model.zerograds()
            loss.backward()
            optimizer.update()
            s = []                
        if (pos % 100 == 0):
            print pos, "/", len(train_data)," finished"
    outfile = "myrnn-" + str(epoch) + ".model"
    serializers.save_npz(outfile, model)

