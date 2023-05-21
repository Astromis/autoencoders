import torch
import torch.nn as nn

class ClusteringLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer. Used in DCEC """
    def __init__(self, size_in, n_clusters, alpha=1.0, **kwargs ):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha        
        
        self.size_in=size_in 
        weights = torch.Tensor(self.n_clusters, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        nn.init.xavier_uniform_(self.weights) # weight init

    def init_weights(self, init_clusters):
        self.weights = nn.Parameter(init_clusters)
    
    def forward(self, inputs):
        q = 1.0 / (1.0 + (torch.sum(torch.square(torch.unsqueeze(inputs, dim=1) - self.weights), dim=2) / self.alpha))
        q = torch.pow(q, (self.alpha + 1.0) / 2.0 ) 
        q = torch.transpose(torch.transpose(q, 0, 1) / torch.sum(q, dim=1), 0,1)
        return q