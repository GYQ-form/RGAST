import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.nn.conv.rgat_conv import RGATConv

class RGAST(torch.nn.Module):
    def __init__(self, hidden_dims, att_drop, dim_reduce='PCA'):
        super(RGAST, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = RGATConv(in_dim, num_hidden, num_relations=2, heads=1, concat=False,
                              dropout=att_drop, add_self_loops=False, bias=False)
        self.conv2 = RGATConv(num_hidden, out_dim, num_relations=2, heads=1, concat=False,
                              dropout=att_drop, add_self_loops=False, bias=False)
        if dim_reduce=='PCA':
            self.decoder = nn.Sequential(
                nn.Linear(out_dim, num_hidden),
                nn.Linear(num_hidden, in_dim),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(out_dim, num_hidden),
                nn.Linear(num_hidden, in_dim),
                nn.ReLU()
            )                  

    def forward(self, features, edge_index, edge_type):
        h1, att1 = self.conv1(features, edge_index, edge_type, return_attention_weights=True)
        h1 = F.elu(h1)
        h2, att2 = self.conv2(h1, edge_index, edge_type, return_attention_weights=True)
        h2 = F.elu(h2)
        h3 = self.decoder(h2)
        return h2, h3, att1, att2

    
class RGAST2(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(RGAST2, self).__init__()
        [in_dim, num_hidden1, num_hidden2, out_dim] = hidden_dims
        self.conv1 = RGATConv(in_dim, num_hidden1, num_relations=2, heads=1, concat=False,
                              dropout=0.3, add_self_loops=False, bias=False)
        self.conv2 = RGATConv(num_hidden1, num_hidden2, num_relations=2, heads=1, concat=False,
                              dropout=0.3, add_self_loops=False, bias=False)
        self.conv3 = RGATConv(num_hidden2, out_dim, num_relations=2, heads=1, concat=False,
                              dropout=0.3, add_self_loops=False, bias=False)                      
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, num_hidden2),
            nn.ReLU(), 
            nn.Linear(num_hidden2, num_hidden1),
            nn.ReLU(), 
            nn.Linear(num_hidden1, in_dim),
        )                  

    def forward(self, features, edge_index, edge_type):
        h1 = F.elu(self.conv1(features, edge_index, edge_type))
        h2 = F.elu(self.conv2(h1, edge_index, edge_type))
        h3 = F.elu(self.conv3(h2, edge_index, edge_type))
        out = self.decoder(h3)
        return h3, out


class adaptive_RGAST(torch.nn.Module):
    def __init__(self, x_dim, base_model, base_dim):
        super(adaptive_RGAST, self).__init__()
        self.linear1 = nn.Linear(x_dim, base_dim)
        self.base_RGAST = base_model
        self.linear2 = nn.Linear(base_dim, x_dim)

    def forward(self, features, edge_index, edge_type):
        h1 = F.relu(self.linear1(features))
        latent, h2 = self.base_RGAST(h1, edge_index, edge_type)
        h3 = self.linear2(F.relu(h2))
        return latent, h3