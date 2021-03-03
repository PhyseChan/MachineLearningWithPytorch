import torch as tr
import math

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
        return x + a, a + x


class RNN(tr.nn.Module):
    def __init__(self, input_size, output_size, model='rnn_cell'):
        super(RNN, self).__init__()
        if model == 'rnn_cell':
            self.rnn_cell = RNN_cell(input_size, output_size)
        elif model == 'rnn_cell_no_module':
            self.rnn_cell = RNN_cell_no_module(input_size, output_size)
        else:
            raise AttributeError

    def forward(self, x, a=None):
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
        return xi, a


class RNN_cell_no_module(tr.nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN_cell_no_module, self).__init__()
        self.output_size = output_size
        std = 1.0 / math.sqrt(output_size)
        self.wa = tr.nn.Parameter(tr.nn.init.uniform_(tr.Tensor(output_size, output_size), -std, std))
        self.wx = tr.nn.Parameter(tr.nn.init.uniform_(tr.Tensor(output_size, input_size), -std, std))
        self.b = tr.nn.Parameter(tr.nn.init.uniform_(tr.Tensor(output_size), -std, std))

    def para_init(self):
        stdv = 1.0 / math.sqrt(self.output_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, a=None):
        x = x @ self.wx.T + self.b
        a = a @ self.wa.T
        return x + a, x + a
