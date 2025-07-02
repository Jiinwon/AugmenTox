import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, heads=1, dropout=0.5):
        super(GATNet, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.heads = heads
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim * heads if i != 0 else hidden_dim
            out_dim = hidden_dim
            # Use heads for all layers except possibly the last
            if i < num_layers - 1:
                conv = GATConv(in_dim, out_dim, heads=heads, concat=True, dropout=dropout)
            else:
                # Last layer: use one head (no concatenation) for output stability
                conv = GATConv(in_dim, out_dim, heads=1, concat=True, dropout=dropout)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(out_dim * (heads if i < num_layers - 1 else 1)))
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = global_mean_pool(x, batch)
        out = self.fc(x)
        return out
    def get_graph_embedding(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        return x
