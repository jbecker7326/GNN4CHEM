import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class GCNConvNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=6):
        super(GCNConvNet, self).__init__()

        self.conv1 = SAGEConv(in_channels, out_channels=128, aggr="max")
        self.conv2 = SAGEConv(in_channels=128, out_channels=128, aggr="max")
        self.conv3 = SAGEConv(in_channels=128, out_channels=128, aggr="max")
        self.dropout = nn.Dropout(0.25)
        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):

        x, edge_index, batch = batch.x, batch.edge_index, batch.batch

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)

        x = torch.relu(self.lin1(x))
        x = self.dropout(x)
        x = torch.relu(self.lin2(x))
        x = self.lin3(x)
        x = self.sigmoid(x)
        return x
