import time
import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
class GCNConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        super(GCNConvNet, self).__init__()
        self.gcn_0 = GCNConv(in_channels, 64)
        self.gcn_h1 = GCNConv(64, 64)
        self.gcn_h2 = GCNConv(64, 64)
        self.gcn_h3 = GCNConv(64, 64)
        self.gcn_h4 = GCNConv(64, 64)
        self.gcn_h5 = GCNConv(64, 64)
        self.gcn_h6 = GCNConv(64, 64)
        self.gcn_out = GCNConv(64, out_channels)
    def forward(self, batch):
        x, edge_index, batch_graph = batch.x, batch.edge_index, batch.batch
        x = torch.relu(self.gcn_0(x, edge_index))
        x = torch.relu(self.gcn_h1(x, edge_index))
        x = torch.relu(self.gcn_h2(x, edge_index))
        x = torch.relu(self.gcn_h3(x, edge_index))
        x = torch.relu(self.gcn_h4(x, edge_index))
        x = torch.relu(self.gcn_h5(x, edge_index))
        x = torch.relu(self.gcn_h6(x, edge_index))
        x = torch.dropout(x, p=0.25, train=self.training)
        x = self.gcn_out(x, edge_index)
        x = torch.sigmoid(x)
        return x
