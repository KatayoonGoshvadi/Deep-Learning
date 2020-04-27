import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import numpy as np


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class FlowLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(FlowLSTM, self).__init__()
        # build your model here
        # your input should be of dim (batch_size, seq_len, input_size)
        # your output should be of dim (batch_size, seq_len, input_size) as well
        # since you are predicting velocity of next step given previous one
        
        # feel free to add functions in the class if needed
        
        self.hidden_size = hidden_size
        self.input_size  = input_size
        self.num_layers  = num_layers

        if   num_layers == 2:
            self.lstm1 = nn.LSTMCell(self.input_size, self.hidden_size)
            self.lstm2 = nn.LSTMCell(self.hidden_size,self.hidden_size)
        elif num_layers == 2:
            self.lstm1 = nn.LSTMCell(self.input_size, self.hidden_size)

        self.linear = nn.Linear(self.hidden_size, self.input_size)




    # forward pass through LSTM layer
    def forward(self, x):
       
        '''
        input: x of dim (batch_size, 19, 17)
        '''
        # define your feedforward pass
        self.seqSize = x.shape[1]
        outputs = torch.empty(x[:,0:1,:].shape).to(device)

        if self.num_layers == 1:
            h_t  = torch.zeros(x.shape[0] , self.hidden_size).to(device) 
            c_t  = torch.zeros(x.shape[0] , self.hidden_size).to(device)
        elif self.num_layers == 2:
            h_t   = torch.zeros(x.shape[0] , self.hidden_size).to(device)
            c_t   = torch.zeros(x.shape[0] , self.hidden_size).to(device)
            h_t2  = torch.zeros(x.shape[0] , self.hidden_size).to(device)
            c_t2  = torch.zeros(x.shape[0] , self.hidden_size).to(device)

        if self.num_layers == 1:
            for i in range(x.shape[1]):
                h_t, c_t = self.lstm1(x[:,i,:], (h_t, c_t))
                output = self.linear(h_t)
                outputs = torch.cat((outputs, output.unsqueeze(1)), 1)


        elif self.num_layers == 2:
            for i in range(x.shape[1]):
                h_t, c_t = self.lstm1(x[:,i,:], (h_t, c_t))
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                output = self.linear(h_t2)
                outputs = torch.cat((outputs, output.unsqueeze(1)), 1)

        return outputs[:,1:,:]


        


    # forward pass through LSTM layer for testing
    def test(self, x):
        '''
        input: x of dim (batch_size, 17)
        '''
        # define your feedforward pass

        outputs = torch.empty(x.unsqueeze(1).shape).to(device)

        if self.num_layers == 1:
            h_t  = torch.zeros(x.shape[0] , self.hidden_size).to(device) 
            c_t  = torch.zeros(x.shape[0] , self.hidden_size).to(device)
        elif self.num_layers == 2:
            h_t   = torch.zeros(x.shape[0] , self.hidden_size).to(device)
            c_t   = torch.zeros(x.shape[0] , self.hidden_size).to(device)
            h_t2  = torch.zeros(x.shape[0] , self.hidden_size).to(device)
            c_t2  = torch.zeros(x.shape[0] , self.hidden_size).to(device)
       

        if self.num_layers == 1:
            for i in range(self.seqSize):
                h_t, c_t = self.lstm1(x, (h_t, c_t))
                x = self.linear(h_t)
                outputs = torch.cat((outputs, x.unsqueeze(1)), 1)

        elif self.num_layers == 2:
            for i in range(self.seqSize):
                h_t, c_t = self.lstm1(x, (h_t, c_t))
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                x = self.linear(h_t2)
                outputs = torch.cat((outputs, x.unsqueeze(1)), 1)

        return outputs[:,1:,:]


















