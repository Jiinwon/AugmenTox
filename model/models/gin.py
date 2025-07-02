import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

class GINNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super(GINNet, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        # Build multiple GIN convolution layers
        for i in range(num_layers):
            if i == 0:
                nn_linear = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            else:
                nn_linear = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            conv = GINConv(nn_linear)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        # Final linear layer for output (graph classification)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # GIN layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        # Global pooling (mean pooling)
        x = global_mean_pool(x, batch)
        out = self.fc(x)
        return out
    def get_graph_embedding(self, data):
        """
        Returns the graph embedding (after pooling, before final output layer).
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            # Do not apply dropout for embedding extraction to get deterministic results
        x = global_mean_pool(x, batch)
        return x
