import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
#epoch 100, learning rate 0.005
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
class GCNConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        super(GCNConvNet, self).__init__()

        self.gat_conv1 = GATConv(in_channels, in_channels)
        self.gat_conv2 = GATConv(in_channels, in_channels,heads=4, dropout=0.0)
        self.fc = nn.Linear(in_channels*4, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0)
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, out_channels)
    def forward(self, batch):
        x, edge_index, batch_graph = batch.x, batch.edge_index, batch.batch
        x = torch.relu(self.gat_conv1(x, edge_index))
        x = torch.relu(self.gat_conv2(x, edge_index))
        x = self.relu(self.fc(x))
        x=self.bn1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x