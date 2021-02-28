import torch as tr
from RNN import RNN as Rnn


class MyModel(tr.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.rnn = Rnn(input_size, hidden_size)
        #self.rnn = tr.nn.RNN(input_size,hidden_size)
        self.fc = tr.nn.Linear(hidden_size, output_size)

    def forward(self, x, a):
        x, a = self.rnn(x, a)
        x = self.fc(x)
        return x, a
