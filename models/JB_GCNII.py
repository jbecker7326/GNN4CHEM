import torch
import torch.nn as nn
import copy
from torch_geometric.nn import ClusterGCNConv
from torch_geometric.nn import PointTransformerConv
from torch_geometric.nn import GCN2Conv
import torch.nn.functional as F

class GCNConvNet(nn.Module):
    def __init__(self, in_channels, nlayers, nhidden, out_channels, dropout, alpha):
        super(GCNConvNet, self).__init__()
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.alpha=alpha
        
        self.drop=torch.nn.Dropout(p=dropout)
        self.dropout=dropout
        
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GCN2Conv(channels=nhidden, alpha=self.alpha,add_self_loops=False))
        
        self.fc1 = nn.Linear(in_channels, nhidden)
        self.fc2 = nn.Linear(nhidden, out_channels)

        self.soft = nn.LogSoftmax(dim=-1)

    def forward(self, batch):
        x, edge_index, batch_graph = batch.x, batch.edge_index, batch.batch
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act_fn(self.fc1(x))
        x_0 = []
        x_0.append(x)

        for i,con in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.act_fn(con(x,x_0[0],edge_index))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        #x = self.soft(x)
    
        
        return x
