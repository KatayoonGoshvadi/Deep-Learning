import torch
import torch.nn as nn
from torch.nn import functional as F


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, 1)
        # since discriminator is a binary classifier

        self.input_dim = input_dim

        self.l1   = nn.Linear(self.input_dim , 100)
        self.l2   = nn.Linear(100, 128)
        self.l3   = nn.Linear(128, 64)
        self.l4   = nn.Linear(64, 1)
    
    def forward(self, x):
        # define your feedforward pass

        l1  = F.relu( self.l1(x) )
        l2  = F.relu( self.l2(l1) )
        l3  = F.relu( self.l3(l2) )
        out = torch.sigmoid( self.l4(l3) )
        return out

class Generator(nn.Module):
    def __init__(self, latent_dim, airfoil_dim):
        super(Generator, self).__init__()
        # build your model here
        # your output should be of dim (batch_size, airfoil_dim)
        # you can use tanh() as the activation for the last layer
        # since y coord of airfoils range from -1 to 1

        self.latent_dim = latent_dim
        self.output_dim = airfoil_dim

        self.l1  = nn.Linear(self.latent_dim, 64)
        self.l2  = nn.Linear(64, 128)
        self.l3  = nn.Linear(128, 100)
        self.l4  = nn.Linear(100, self.output_dim)
    
    def forward(self, x):
        # define your feedforward pass
        
        l1 = F.relu( self.l1(x) )
        l2 = F.relu( self.l2(l1) )
        l3 = F.relu( self.l3(l2) )
        l4 = self.l4(l3)
        out= torch.tanh(l4)
        return out

