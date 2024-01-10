import  torch
from    torch import nn
from    torch.nn import functional as F
from    utils import sparse_dropout, dot
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class GraphConvolution(nn.Module):


    def __init__(self, input_dim, output_dim,
                 dropout=0.5,
                 is_sparse_inputs=False,
                 bias=False,
                 activation = F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()


        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless


        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight.data, gain=1)
        #self.bias = None
        #if bias:
        self.bias = nn.Parameter(torch.zeros(input_dim, output_dim))
        nn.init.xavier_uniform_(self.bias.data, gain=1)

    def forward(self,inputs):
        # print('inputs:', inputs)
        x, support = inputs
        # dropout

        #x = F.dropout(x, self.dropout)

        # convolve
        if not self.featureless: # if it has features x
            if self.is_sparse_inputs:
                x = x.to(torch.float32)

                xw = torch.sparse.mm(x, self.weight)
            else:
                xw = torch.mm(x, self.weight)
        else:
            xw = self.weight
        support = support.cuda()

        out = torch.mm(support, xw)
        mean, std, var = torch.mean(out), torch.std(out), torch.var(out)
        out = (out - mean) / std
        #if self.bias is not None:
        #    out += self.bias

        return self.activation(out)

