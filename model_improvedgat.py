import torch
from torch_geometric.nn import GATConv
import torch.nn as nn
from torch.nn import functional as F

class ImprovedGAT(torch.nn.Module):
    def __init__(self, encoder_layer, num_layers, hidesize):
        super(ImprovedGAT, self).__init__()
        layer = []
        for l in range(num_layers):
            layer.append(encoder_layer)
        self.layers = nn.ModuleList(layer)
        self.out_mlp = nn.Linear((num_layers+1)*hidesize, hidesize)
    def forward(self, features, edge_index):
        out = features
        output = [out]
        for mod in self.layers:
            out = mod(out, edge_index)
            output.append(out)
        output_ = torch.cat(output, dim=-1)
        output_ = self.out_mlp(output_)
        return output_

class ImprovedGATLayer(torch.nn.Module):
    def __init__(self, hidesize, dropout=0.5, num_heads=5, use_residual=True, no_cuda=False):
        super(ImprovedGATLayer, self).__init__()
        self.no_cuda = no_cuda
        self.use_residual = use_residual
        self.convs = GATConv(hidesize, hidesize, heads=num_heads, add_self_loops=True, concat=False)

    def forward(self, features, edge_index):
        x = features
        if self.use_residual:
            x = x + self.convs(x, edge_index)
        else:
            x = self.convs(x, edge_index)

        return x
