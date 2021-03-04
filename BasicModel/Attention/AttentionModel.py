import torch as tr
import torch.nn.functional as f


class LstmAttentionOnData(tr.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LstmAttentionOnData, self).__init__()
        self.sequenceWeight = tr.nn.Linear(input_size, 1)
        self.condictionWeight = tr.nn.Linear(hidden_size, hidden_size)
        self.lstm = tr.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.fc = tr.nn.Linear(hidden_size, output_size)
        self.H = hidden_size

    def forward(self, x, a):
        """
            B: batch size
            N: the size of features
            S: the length of the sequence
            H: hidden size
            a: shape of (B, H)
            x: shape of (S, B, N)
        """

        # Apply attention mechanism to sequence data
        x_prob = f.softmax(self.sequenceWeight(x), dim=0)  # (S, B, 1)
        x = tr.mul(x, x_prob)  # (S, B, N)

        # Apply attention mechanism to the LSTM output
        x, a = self.lstm(x)  # (S, B, H)

        # full connected layer
        x = self.fc(x[-1])
        return x


class LstmAttentionOnLstm(tr.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LstmAttentionOnLstm, self).__init__()
        self.sequenceWeight = tr.nn.Linear(input_size, 1)
        self.condictionWeight = tr.nn.Linear(hidden_size, hidden_size)
        self.lstm = tr.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.fc = tr.nn.Linear(hidden_size, output_size)
        self.H = hidden_size
        self.batchnorm = tr.nn.BatchNorm1d(hidden_size)

    def forward(self, x, a):
        """
            B: batch size
            N: the size of features
            S: the length of the sequence
            H: hidden size
            a: shape of (B, H)
            x: shape of (S, B, N)
        """
        # Apply attention mechanism to the LSTM output
        x, a = self.lstm(x)  # (S, B, H)
        con_prb = self.condictionWeight(x[-1])  # ( B, H)
        con_prb = f.softmax(con_prb, dim=1)
        x = tr.mul(x[-1], con_prb)  # (B, H)
        x = self.batchnorm(x)

        # full connected layer
        x = self.fc(x)
        return x


class LstmAttention(tr.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LstmAttention, self).__init__()
        self.sequenceWeight = tr.nn.Linear(input_size, 1)
        self.condictionWeight = tr.nn.Linear(hidden_size, hidden_size)
        self.lstm = tr.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.fc = tr.nn.Linear(hidden_size, output_size)
        self.H = hidden_size
        self.batchnorm_1 = tr.nn.BatchNorm1d(input_size)
        self.batchnorm_2 = tr.nn.BatchNorm1d(hidden_size)

    def forward(self, x, a):
        """
            B: batch size
            N: the size of features
            S: the length of the sequence
            H: hidden size
            a: shape of (B, H)
            x: shape of (S, B, N)
        """

        # Apply attention mechanism to sequence data
        x_prob = f.softmax(self.sequenceWeight(x), dim=0)  # (S, B, 1)
        x = tr.mul(x, x_prob)  # (S, B, N)
        x = x.permute([1, 0, 2])
        x = self.batchnorm_1(x)
        x = x.permute([1, 0, 2])

        # Apply attention mechanism to the LSTM output
        x, a = self.lstm(x)  # (S, B, H)
        con_prb = self.condictionWeight(x[-1])  # ( B, H)
        con_prb = f.softmax(con_prb, dim=1)
        x = tr.mul(x[-1], con_prb)  # (B, H)
        x = self.batchnorm_2(x)

        # full connected layer
        x = self.fc(x)
        return x
