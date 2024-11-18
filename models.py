import numpy as np
import torch 
import torch.nn as nn
from scipy import stats as st




class MLP(torch.nn.Module):
    def __init__(self,layers,in_featuers,dropout_rate):
        super(MLP,self).__init__()


 
        self.num_layers = len(layers)
        self.layers = nn.ModuleList()
         
        self.layers.append(nn.Linear(in_featuers,layers[0]))
        for i in range(self.num_layers-1) :
            self.layers.append(nn.Linear(layers[i],layers[i+1]))
        self.out_layer = nn.Linear(layers[-1],1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self,x):
    
        for i,layer in enumerate(self.layers):
            x = self.activation(layer(x))
        x = self.drop(x)
        out = self.out_layer(x)


        return self.sigmoid(out)
    


class RaGrd():
    def __init__(self,k):
        self.k = k
       
    
    def assign_clusters(self,data,centroids):
        norm = np.linalg.norm(np.expand_dims(data,axis=1)-centroids,axis=2) # calculate norm on last dim (num_data,centroids,dim)
        return np.argsort(norm,axis=1)
        
    def predict(self,data,centroids,centroids_label):
        result = self.assign_clusters(data,centroids)
        nearest_k = result[:,:self.k]
        
        nearest_labels = np.take(centroids_label,nearest_k)
        
        preds = st.mode(nearest_labels,axis=1).mode
        
        return preds

    









