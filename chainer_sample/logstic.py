#!/usr/bin/env python

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable 
from chainer import optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

# Set data

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data.astype(np.float32)
Y = iris.target.astype(np.float32)
N = Y.size
Y2 = np.zeros(3 * N).reshape(N,3).astype(np.float32)
for i in range(N):
    Y2[i,Y[i]] = 1.0

index = np.arange(N)
xtrain = X[index[index % 2 != 0],:]
ytrain = Y2[index[index % 2 != 0],:]
xtest = X[index[index % 2 == 0],:]
yans = Y[index[index % 2 == 0]]

# Define model

class IrisRogi(Chain):
    def __init__(self):
        super(IrisRogi, self).__init__(
            l1=L.Linear(4,3),
        )

    def __call__(self,x,y):
        return F.mean_squared_error(self.fwd(x), y)

    def fwd(self,x):
        return F.softmax(self.l1(x))

# Initialize model

model = IrisRogi()
optimizer = optimizers.Adam()
optimizer.setup(model)

# Learn

n = 75    
bs = 25   
for j in range(5000):   
    sffindx = np.random.permutation(n)
    for i in range(0, n, bs):
        x = Variable(xtrain[sffindx[i:(i+bs) if (i+bs) < n else n]])
        y = Variable(ytrain[sffindx[i:(i+bs) if (i+bs) < n else n]])
        model.zerograds()
        loss = model(x,y)
        loss.backward()
        optimizer.update()

# Test

xt = Variable(xtest, volatile='on')
yy = model.fwd(xt)

ans = yy.data
nrow, ncol = ans.shape
ok = 0
for i in range(nrow):
    cls = np.argmax(ans[i,:])
    print ans[i,:], cls            
    if cls == yans[i]:
        ok += 1
        
print ok, "/", nrow, " = ", (ok * 1.0)/nrow

