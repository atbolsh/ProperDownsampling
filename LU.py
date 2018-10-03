"""
This is the module for the LU layer
It acts like a linear layer, but stores the weights in an LU matrix.
"""

#import nln

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


class LU(nn.Module):
    def __init__(self, n, bias=True):
        super(LU, self).__init__()

        self.n = n
        #Top triangle is U, including diagonal; bottom is L, excluding diagonal
        #Written this way to save space.
        self.weight    = Parameter(torch.Tensor(n, n))
        self.mask      = torch.eye(n)
        if bias:
            self.bias = Parameter(torch.Tensor(n))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
   
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        #This constructs the LU decomposition. Hacky; rewrite.
#        _, self.weight.data = torch.gesv(torch.zeros(self.n), self.weight.data)
        #Now, fix the diagonal. Probably a faster way to do this . . .
        #For the record: idea is to make sure that the diagonal has some variation, 
        #but the expected value of the determinant is still 1
        self.weight.data *= (1 - torch.eye(self.n))
#        A = torch.Tensor(self.n)
#        c = math.log(2) #Make this a less magic number
#        A.uniform_(-c, c)
        #REALLY fix this
#        A[-1] = 0 - torch.sum(A[:-1])
        self.weight.data += torch.eye(self.n)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
   
    def logJ(self):
        return torch.sum(torch.log(torch.abs(torch.diag(self.weight))))
       
    def forward(self, x):
        U = self.weight.triu()
        mid = x.matmul(U.t())
        #Simulates L = bottom half + Id matrix. Reason is bugs with transferring onto GPU.
        y = self.bias + mid.matmul((self.weight - U).t()) + mid
        #Givew result and log Jcobian
        #J = Sum of log of local derivative and the log of the diagonal of the matrix.
        return y, self.logJ()
    
    def extra_repr(self):
        return 'n={}, bias={}'.format( \
            self.n, self.bias is not None \
        )

    def pushback(self, y):
        #Solves LU x + bias = y, and the current forward logJ
        #Get that forward logJ
        #J = Sum of log of local derivative and the log of the diagonal of the matrix.
        mid = y - self.bias.detach()
        J = self.logJ()
        #No avoiding sequentialism, unfortunately
        for i in range(1, self.n): 
            mid[:, i]  -= mid[:, :i].matmul(self.weight[i, :i].detach())
        #U x = mid
        x = mid/torch.diag(self.weight).detach()
        for i in range(2, self.n + 1):
            x[:, -i] -= x[:, 1-i:].matmul(self.weight[-i, 1-i:].detach())/self.weight[-i, -i].detach()
        return x, J



    
