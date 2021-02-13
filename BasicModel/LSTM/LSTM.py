import torch as tr
import torch.nn as nn
import torch.nn.functional as F

class LSTM_cell(nn.Module):
    def __init__(self, input_size, output_size, h_size):
        c_size = output_size
        super(LSTM_cell, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(input_size+h_size, c_size),
            nn.Sigmoid(),
        )
        self.i = nn.Sequential(
            nn.Linear(input_size+h_size, c_size),
            nn.Sigmoid(),
        )
        self.o = nn.Sequential(
            nn.Linear(input_size + h_size, c_size),
            nn.Sigmoid(),
        )
        self.c_h = nn.Sequential(
            nn.Linear(input_size + h_size, c_size),
            nn.Tanh(),
        )

    def forward(self, x, h, c):
        x = tr.cat(x, h, dim=0)
        f = self.f(x)
        i = self.i(x)
        c_h = self.c_h(x)
        o = self.o(x)
        h_next = f * c + i * c_h
        c_next = F.tanh(h_next) * o
        return h_next, c_next

class LSTM(nn.Module):