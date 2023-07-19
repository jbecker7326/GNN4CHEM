import torch
import torch.nn as nn
from torch_geometric.nn import ClusterGCNConv

class GCNConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        super(GCNConvNet, self).__init__()
        self.cluster_conv = ClusterGCNConv(in_channels, in_channels)
        self.fc = nn.Linear(in_channels, 128)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, out_channels)
    def forward(self, batch):
        x, edge_index, batch_graph = batch.x, batch.edge_index, batch.batch
        x = torch.relu(self.cluster_conv(x, edge_index))
        x = self.relu(self.fc(x))
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x
