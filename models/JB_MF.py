import torch
import torch.nn as nn
from torch_geometric.nn import MFConv

class GCNConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        super(GCNConvNet, self).__init__()
        self.conv1 = MFConv(in_channels, 4*in_channels, alpha=0.95)
        self.seq1 = nn.Sequential(nn.BatchNorm1d(4 * in_channels),
                          nn.ReLU(),
                          nn.Linear(4 * in_channels, 4 * in_channels),
                          nn.ReLU())
        
        self.conv2 = MFConv(4*in_channels, 8*in_channels, alpha=0.95)
        self.seq2 = nn.Sequential(nn.BatchNorm1d(8 * in_channels),
                          nn.ReLU(),
                          nn.Linear(8 * in_channels, 8 * in_channels),
                          nn.ReLU())
        
        self.conv3 = MFConv(8*in_channels, 16*in_channels, alpha=0.95)
        self.seq3 = nn.Sequential(nn.BatchNorm1d(16 * in_channels),
                          nn.ReLU(),
                          nn.Linear(16 * in_channels, 16 * in_channels),
                          nn.ReLU())
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(16*in_channels, 8*in_channels)
        self.linear2 = nn.Linear(8*in_channels, 4*in_channels)
        self.out = nn.Linear(4*in_channels, out_channels)
    def forward(self, batch):
        x, edge_index, batch_graph = batch.x, batch.edge_index, batch.batch
        x = self.seq1(self.conv1(x, edge_index))
        x = self.seq2(self.conv2(x, edge_index))
        x = self.seq3(self.conv3(x, edge_index))
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x
