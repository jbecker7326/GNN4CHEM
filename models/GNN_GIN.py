import torch
import torch.nn as nn
from torch_geometric.nn import GINConv

class GCNConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        super(GCNConvNet, self).__init__()
        self.conv1 = GINConv(
            nn.Sequential(nn.Linear(in_channels, 4 * in_channels),
                          nn.BatchNorm1d(4 * in_channels),
                          nn.ReLU(),
                          nn.Linear(4 * in_channels, 4 * in_channels),
                          nn.ReLU())
        )
        self.conv2 = GINConv(
            nn.Sequential(nn.Linear(in_channels, 4 * in_channels),
                          nn.BatchNorm1d(4 * in_channels),
                          nn.ReLU(),
                          nn.Linear(4 * in_channels, 4 * in_channels),
                          nn.ReLU())
        )
        self.fc = nn.Linear(8 * in_channels, 16 * in_channels)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(16 * in_channels, 32 * in_channels)
        self.linear2 = nn.Linear(32 * in_channels, 16 * in_channels)
        self.out = nn.Linear(16 * in_channels, out_channels)
    def forward(self, batch):
        x, edge_index, batch_graph = batch.x, batch.edge_index, batch.batch
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x, edge_index)
        x = torch.cat((x1, x2), dim = 1)
        x = self.relu(self.fc(x))
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x
