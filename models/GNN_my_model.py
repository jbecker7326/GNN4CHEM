import torch
import torch.nn as nn
from torch_geometric.nn import ClusterGCNConv

class GCNConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        super(GCNConvNet, self).__init__()
        self.cluster_conv1 = ClusterGCNConv(in_channels, 64)
        self.cluster_conv2 = ClusterGCNConv(64, 128)
        self.cluster_conv3 = ClusterGCNConv(128, 256)
        self.linear = nn.Linear(256, out_channels)
    def forward(self, batch):
        x, edge_index, batch_graph = batch.x, batch.edge_index, batch.batch
        x = torch.relu(self.cluster_conv1(x, edge_index))
        x = torch.relu(self.cluster_conv2(x, edge_index))
        x = torch.relu(self.cluster_conv3(x, edge_index))
        x = torch.dropout(x, p=0.25, train=self.training)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x
