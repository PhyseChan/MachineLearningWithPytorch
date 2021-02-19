import torch as tr
import torch.nn as nn
import torch.nn.functional as F


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
        f = F.sigmoid(self.wfh(h) + self.wfin(x))
        i = F.sigmoid(self.wih(h) + self.wiin(x))
        c_ = F.tanh(self.wch(h) + self.wcin(x))
        o = F.sigmoid(self.woh(h) + self.woin(x))
        c_next = f * c + i * c_
        h_next = F.tanh(c_next) * o

        return h_next, c_next


class LSTM(nn.Module):
    # def __init__(self, in_size, out_size):
    #     super(LSTM, self).__init__()
    #     self.lstm_cell_1 = LSTM_cell(in_size, out_size)
    #     self.lstm_cell_2 = LSTM_cell(in_size, out_size)
    #
    # def forward(self, x, h0, c0):
    #     sequence_len = x.shape[1]
    #     c = c0
    #     y = h0
    #     for i in range(sequence_len):
    #         y, c = self.lstm_cell_1(x[:, i, :], y, c)
    #     return y, c
    def __init__(self, in_size,hidden_size, out_size):
        super(LSTM, self).__init__()
        self.lstm_cell_1 = nn.RNN(in_size, hidden_size,2)
        self.fc = nn.Linear(hidden_size, out_size )

    def forward(self, x, h0, c0):
        sequence_len = x.shape[1]
        c = c0
        y = h0
        x = x.permute(1,0,2)
        y, c = self.lstm_cell_1(x)
        y = self.fc(y[-1,:,:])
        return y

