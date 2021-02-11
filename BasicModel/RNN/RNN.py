import torch as tr
import numpy as np


class RNN(tr.nn.Module):
    def __init__(self, input_size, output_size, a0):
        super(RNN, self).__init__()
        self.wa = tr.nn.Linear(output_size, output_size)
        self.wx = tr.nn.Linear(input_size, output_size)
        self.a0 = a0

    def forward(self, x):
        y1 = self.wx(x[:,:,0]) + self.wa(self.a0)
        y1 = tr.nn.functional.relu(y1)
        y2 = self.wx(x[:,:,1]) + self.wa(y1)
        y2 = tr.nn.functional.relu(y2)
        y3 = self.wx(x[:,:,2]) + self.wa(y2)
        y3 = tr.nn.functional.relu(y3)
        y4 = self.wx(x[:,:,3]) + self.wa(y3)
        y4 = tr.nn.functional.relu(y4)
        y5 = self.wx(x[:,:,4]) + self.wa(y4)
        return y5
# class RNN(tr.nn.Module):
#     def __init__(self, input_size, output_size, a0):
#         super(RNN, self).__init__()
#         self.rnn = tr.nn.RNN(input_size,hidden_size=output_size, num_layers=1,nonlinearity='relu')
#
#     def forward(self, x):
#        output, h_0 = self.rnn(x,)
#        return output