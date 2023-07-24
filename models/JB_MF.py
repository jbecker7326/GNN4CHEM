import torch
import torch.nn as nn
from torch_geometric.nn import MFConv

class GCNConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        super(GCNConvNet, self).__init__()
        self.conv1 = MFConv(in_channels, 128, alpha=0.95)
        self.fc1 = nn.Linear(128, 164)
        self.conv2 = MFConv(164, 286, alpha=0.95)
        self.fc2 = nn.Linear(286, 360)
        self.conv3 = MFConv(360, 286, alpha=0.95)
        self.fc3 = nn.Linear(286, 164)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(164, 128)
        self.linear2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, out_channels)
    def forward(self, batch):
        x, edge_index, batch_graph = batch.x, batch.edge_index, batch.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = self.relu(self.fc1(x))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.relu(self.fc2(x))
        x = torch.relu(self.conv3(x, edge_index))
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x
