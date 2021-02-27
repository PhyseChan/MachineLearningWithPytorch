import torch as tr
import torch.nn as nn


class LSTM_cell(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTM_cell, self).__init__()
        self.wfh = nn.Linear(output_size, output_size)
        self.wfin = nn.Linear(input_size, output_size)

        self.wih = nn.Linear(output_size, output_size)
        self.wiin = nn.Linear(input_size, output_size)

        self.wch = nn.Linear(output_size, output_size)
        self.wcin = nn.Linear(input_size, output_size)

        self.woh = nn.Linear(output_size, output_size)
        self.woin = nn.Linear(input_size, output_size)

    def forward(self, x, h, c):
        f = tr.sigmoid(self.wfh(h) + self.wfin(x))
        i = tr.sigmoid(self.wih(h) + self.wiin(x))
        c_ = tr.tanh(self.wch(h) + self.wcin(x))
        o = tr.sigmoid(self.woh(h) + self.woin(x))
        c_next = f * c + i * c_
        h_next = tr.tanh(c_next) * o

        return h_next, c_next


class LSTM(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(LSTM, self).__init__()
        self.lstm_cell_1 = LSTM_cell(in_size, hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)
        self.in_size = in_size
        self.hidden_size = hidden_size

    def forward(self, x):
        sequence_len = x.shape[1]
        batch_size = x.shape[0]
        h = tr.zeros(batch_size, 1, self.hidden_size)
        c = tr.randn(batch_size,1, self.hidden_size)
        for i in range(sequence_len):
            h, c = self.lstm_cell_1(x[:, i, :].unsqueeze(1), h, c)
        h = self.linear(h)
        return h



