import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GATConv, GCNConv
from torch.autograd import Variable
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import utils
from Setting import *

class GAT(torch.nn.Module):
    
    def __init__(self):
        super(GAT, self).__init__()
        self.hidden_channels = HIDDEN_CHANNELS
        self.input_channels = INPUT_CHANNELS
        self.headsv1 = 1
        self.headsv2 = 1
        
        self.conv1 = GATConv(in_channels=self.input_channels, out_channels=self.hidden_channels,
                           heads=self.headsv1, dropout=0.2)
        ####
#         self.conv2 = GATv2Conv(in_channels=self.hidden_channels*self.headsv1, out_channels=self.hidden_channels,
#                              heads=self.headsv2, dropout=0.2)
        ####
        
    def forward(self, data, item_len):
        x, edge_index = data.x, data.edge_index  
        x_in = Variable(x, requires_grad=True)
        x = F.dropout(x_in, p=0.2, training=self.training) 
        x = self.conv1(x, edge_index)                   
        x = F.elu(x)
        
        ###
#         x = F.dropout(x, p=0.2, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = F.elu(x)
        ###
        
        y = x[0,:] * x[1:item_len,:]
        y = torch.sum(y, dim=1, dtype=float)
        y = F.elu(y)
        y.retain_grad()
        return x_in, y