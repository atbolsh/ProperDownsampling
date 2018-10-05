"""
Layer for selecting a learnable subspace of m.

Rows are orthogonal, unit vectors.
Bias is subtracted at the start (to move into the right position).

Cool thing about this parametrization: pushback is the Moore-Penrose pseudoinverse.


This is the better way to do things, over old - orthogonality is self-reinforcing, not forced / unstable.
"""

from scipy.linalg import svd


import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from copy import deepcopy
import math
import numpy as np

class down(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(down, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            #Remember, here bias is an offset in DOMAIN, not codomain
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        #Cheap procedure to get a unitary matrix, and preserve only part of it.
        a = torch.randn(self.out_features, self.in_features)
        _, _, start = svd(a.numpy())
        self.weight = Parameter(torch.Tensor(start[:self.out_features, :]))
    
    def rescale(self):
        """Resets rows to 1-norm."""
        n = torch.sqrt(torch.sum(self.weight.data*self.weight.data, 1)).view(self.out_features, 1)
#        print(torch.max(n))
        self.weight.data = self.weight.data/n
#        n = torch.sqrt(torch.sum(self.weight.data*self.weight.data, 1)).view(self.out_features, 1)
#        print(torch.max(n))

    def forward(self, x, bias=True):
        if type(self.bias)!= type(None) and bias:
             return F.linear(x - self.bias, self.weight, None)
        else:
             return F.linear(x, self.weight, None)

    def pushback(self, y, bias=True):
        if bias:
            return F.linear(y, self.weight.t(), self.bias)
        else:
            return F.linear(y, self.weight.t(), None)

    def collapse(self, x, bias=True):
        """Stays in codomain, but goes down to this linear space."""
        return self.pushback(self.forward(x, bias), bias)
    

model = down(30, 10).cuda()

optimizer = optim.SGD(model.parameters(), lr=0.001)
M = torch.tensor([10. for i in range(5)] + [2., 3., 4., 5., 9.] + [12. for i in range(5)] + [3. for i in range(15)]).cuda()

b = torch.tensor([1. for i in range(30)]).cuda()

print(model.weight)

for i in range(1000):
    model.zero_grad()
    x = torch.randn(1000, 30).cuda()*M + b
    y = model.collapse(x)
    z = y - x
    loss = torch.sum(z*z)/1000.
    if i%1 == 0:
        print(loss)
        print(model.weight)
    loss.backward()
    optimizer.step()

"""
model.reOrth()

for i in range(1000):
    print(model.badness())
    model.reOrth()
"""
print(torch.matmul(model.weight.t(), model.weight).cpu().detach().numpy().round(2))
print(model.weight)
print(M)
print(model.bias)
print(b)
