"""
Layer for selecting a learnable subspace of m.

Rows are orthogonal, unit vectors.
Bias is subtracted at the start (to move into the right position).

Cool thing about this parametrization: pushback is the Moore-Penrose pseudoinverse.
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

        def bH(grad):
            with torch.no_grad():
                return grad - self.collapse(grad, bias=False)

        self.weight.register_hook(bH)

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

    def _forward(self, x, bias=True):
        if type(self.bias)!= type(None) and bias:
             return F.linear(x - self.bias, self.weight, None)
        else:
             return F.linear(x, self.weight, None)

    def _pushback(self, y, bias=True):
        if bias:
            return F.linear(y, self.weight.t(), self.bias)
        else:
            return F.linear(y, self.weight.t(), None)

    def forward(self, x, bias=True):
        #Option to ignore offset useful for gradient reset.
        while self.badness() > 1e-4:
            with torch.no_grad():
                self.reOrth()
        return self._forward(x, bias)

    def pushback(self, y, bias=True):
        while self.badness() > 1e-4:
            with torch.no_grad():
                self.reOrth()
        return self._pushback(y, bias)
    
    def collapse(self, x, bias=True):
        """Stays in codomain, but goes down to this linear space."""
        return self._pushback(self._forward(x, bias), bias)

    def reOrth(self):
        #This is a way to push the vectors away from each other.
        #First order reorthogonalizaiton
        self.weight = Parameter(self.weight + (self.weight - self.collapse(self.weight, bias=False))/self.out_features)
        self.rescale()
    
    def badness(self):
        """Measure of non-orthogonality"""
        if self.weight.data.is_cuda:
            y = torch.matmul(self.weight, self.weight.t()) - torch.eye(self.out_features).cuda()
        else:
            y = torch.matmul(self.weight, self.weight.t()) - torch.eye(self.out_features)
        return torch.sum(y*y)

model = down(300, 200).cuda()
optimizer = optim.SGD(model.parameters(), lr=1e-2)

for i in range(1000):
    x = torch.randn(300).cuda()
    y = model(x)
    loss = torch.sum(y)
    loss.backward()
    optimizer.step()
    if i%1 == 0:
        print(torch.matmul(model.weight.data, model.weight.data.t()))
"""
model.reOrth()

for i in range(1000):
    print(model.badness())
    model.reOrth()
"""
print(model.badness())

    
