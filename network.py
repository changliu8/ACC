import torch.nn as nn
import torch


class Pred(nn.Module):
    def __init__(self,feature_size,hidden_size,output_size):
        super(Pred,self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(feature_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2,hidden_size)
        self.output = nn.Linear(hidden_size,output_size)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        result = self.relu(self.fc1(x))
        result = self.relu(self.fc2(result))
        result = self.relu(self.fc3(result))
        result = self.output(result)

        return result