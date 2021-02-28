import torch as tr
import numpy as np


class RNN_cell(tr.nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN_cell, self).__init__()
        self.wa = tr.nn.Linear(output_size, output_size)
        self.wx = tr.nn.Linear(input_size, output_size)

    def forward(self, x, a):
        """
            B: batch size
            N: the size of features
            a: shape of (B, B)
            x: shape of (B, N)
        """
        a = self.wa(a)
        x = self.wx(x)
        return x + a, x


class RNN(tr.nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()
        self.rnn_cell = RNN_cell(input_size, output_size)

    def forward(self, x, a=None):
        res_x = []
        """
            B: batch size
            N: the size of features
            S: the length of the sequence
            a: shape of (B, B)
            x: shape of (S, B, N)
        """
        sequence_len = x.shape[0]
        for i in range(sequence_len):
            xi, a = self.rnn_cell(x[i], a)
            xi = tr.tanh(xi)
            res_x.append(xi)
        return res_x, a
