import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super().__init__()
        self.linear_1 = nn.Linear(n_inputs, n_hidden) 
        self.linear_2 = nn.Linear(n_hidden, n_hidden * 2) 
        self.linear_3 = nn.Linear(n_hidden * 2, n_hidden) 
        self.linear_4 = nn.Linear(n_hidden, n_outputs)
        self.relu = nn.ReLU()
    
    def forward(self, X):
        out = self.linear_1(X)
        out = self.relu(out)
        out = self.linear_2(out)
        out = self.relu(out)
        out = self.linear_3(out)
        out = self.relu(out)
        out = self.linear_4(out)
        return out