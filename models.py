import numpy as np
import torch 
import torch.nn as nn





class MLP(torch.nn.Module):
    def __init__(self,layers,in_featuers,n_classes,dropout_rate):
        super(MLP,self).__init__()


 
        self.num_layers = len(layers)
        self.layers = nn.ModuleList()
         
        self.layers.append(nn.Linear(in_featuers,layers[0]))
        for i in range(self.num_layers-1) :
            self.layers.append(nn.Linear(layers[i],layers[i+1]))
        self.activation = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self,x):

        for i in range(self.num_layers-1):
            x = self.drop(self.activation(self.layers[i](x)))

        out = self.layers[self.num_layers-1](x)

        return out










