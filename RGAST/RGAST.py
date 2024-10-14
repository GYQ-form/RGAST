import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.nn.conv.rgat_conv import RGATConv

class RGAST(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(RGAST, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = RGATConv(in_dim, num_hidden, num_relations=2, heads=1, concat=False,
                              dropout=0.3, bias=False)
        self.conv2 = RGATConv(num_hidden, out_dim, num_relations=2, heads=1, concat=False,
                              dropout=0.3, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, in_dim),
        )                  

    def forward(self, features, edge_index, edge_type):
        h1 = F.elu(self.conv1(features, edge_index, edge_type))
        h2 = F.elu(self.conv2(h1, edge_index, edge_type))
        h3 = self.decoder(h2)
        return h2, h3