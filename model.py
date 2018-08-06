# import torch 
# import torch.nn as nn

# class RNN(nn.Module):
# 	def __init__(self, input_size, hidden_size, output_size):
# 		super(RNN, self).__init__()
# 		self.hidden_size = hidden_size

# 		self.input_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
# 		self.input_to_output = nn.Linear(input_size + hidden_size, output_size) 
# 		self.output_to_output = nn.Linear(output_size + hidden_size, output_size)

# 		self.dropout = nn.Dropout(0.1)
# 		self.softmax = nn.LogSoftmax(dim=0)

# 	def forward(self, input, hidden):
# 		print (input.shape)
# 		print (hidden.shape)
# 		input_combined = torch.cat((input, hidden), 0)
# 		hidden = self.input_to_hidden(input_combined)
# 		output = self.input_to_output(input_combined)
# 		output_combined = torch.cat((hidden, output))
# 		output = self.output_to_output(output_combined)
# 		output = self.dropout(output)
# 		output = self.softmax(output)
# 		return output,hidden

# 	def initHidden(self):
# 		return torch.zeros(self.hidden_size)

import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

