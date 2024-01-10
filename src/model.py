import  torch
from  torch import nn
from  torch.nn import functional as F
from  layer import GraphConvolution



class GCN(nn.Module):


    def __init__(self, input_dim,hidden,output_dim,dropout,adj):
        super(GCN, self).__init__()

        self.input_dim = input_dim # 1433
        self.output_dim = output_dim

        print('input dim:', input_dim)
        print('output dim:', output_dim)

        self.adj = adj

        self.layers1 = nn.Sequential(GraphConvolution(self.input_dim, hidden,
                                                     activation=F.relu,
                                                     dropout=dropout,
                                                     is_sparse_inputs=False),
                                     )
        self.layers2 = nn.Sequential(GraphConvolution(hidden, self.output_dim,
                                                     activation=F.relu,
                                                     dropout=dropout,
                                                     is_sparse_inputs=False),
                                    )

    def forward(self, x):
        #x, support = inputs

        x = self.layers1((x, self.adj))
        x = self.layers2((x, self.adj))
        return x

    def l2_loss(self):

        layer = self.layers.children()
        layer = next(iter(layer))

        loss = None

        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss



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

