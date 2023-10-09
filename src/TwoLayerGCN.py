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
