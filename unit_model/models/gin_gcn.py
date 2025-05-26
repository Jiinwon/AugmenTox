import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from models.gin import GINNet
from models.gcn import GCNNet

class GIN_GCN_Hybrid(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super(GIN_GCN_Hybrid, self).__init__()
        self.gin = GINNet(input_dim, hidden_dim, hidden_dim, num_layers, dropout)
        self.gcn = GCNNet(input_dim, hidden_dim, hidden_dim, num_layers, dropout)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, data):
        h1 = self.gin.get_graph_embedding(data)
        h2 = self.gcn.get_graph_embedding(data)
        h = torch.cat([h1, h2], dim=1)
        out = self.fc(h)
        return out

    def get_graph_embedding(self, data):
        h1 = self.gin.get_graph_embedding(data)
        h2 = self.gcn.get_graph_embedding(data)
        x = torch.cat([h1, h2], dim=1)
        return x
