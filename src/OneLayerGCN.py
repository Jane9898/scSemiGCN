import torch
from  torch import nn
from  torch.nn import functional as F
from  layer import GraphConvolution

class preprocessor(nn.Module):

    def __init__(self,dim,dropout,adj):
        super(preprocessor, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim,512)
        self.linear2 = nn.Linear(dim,dim)
        self.linear3 = nn.Linear(128,32)

        self.dropout = nn.Dropout(p=0.7)
        self.adj = adj
        self.layers = nn.Sequential(GraphConvolution(dim, dim,
                                                 activation=F.relu,
                                                 dropout=dropout,
                                                 is_sparse_inputs=True),)

    def forward(self,x):
    

        x1 = self.layers((x,self.adj))
        x2 = self.linear2(x1)
        x3 = self.dropout(x2)
        #x5 = self.linear3(x4)
        x_ = torch.norm(x3,dim=1,p=2,keepdim=True).detach()
        out = x3/x_


        return out

