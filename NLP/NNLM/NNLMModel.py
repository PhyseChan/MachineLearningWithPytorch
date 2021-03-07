import numpy as np
import torch as tr


class NNLM(tr.nn.Module):
    def __init__(self, vocab_size, hidden_size, repr_size=20, num_words=4):
        super(NNLM, self).__init__()
        self.c = tr.nn.Embedding(vocab_size, repr_size)
        self.h = tr.nn.Parameter(tr.randn(repr_size * num_words, hidden_size))
        self.u = tr.nn.Parameter(tr.randn(hidden_size, vocab_size))
        self.w = tr.nn.Parameter(tr.randn(repr_size * num_words, vocab_size))
        self.b = tr.nn.Parameter(tr.rand(vocab_size))
        self.d = tr.nn.Parameter(tr.rand(hidden_size))
        self.tanh = tr.nn.Tanh()
        self.repr_size = repr_size
        self.num_words = num_words

    def forward(self, x):
        x = self.c(x)
        x = x.view(-1, self.repr_size*self.num_words)  # (1, num_words*repr)
        tanh_x = self.tanh(self.d + tr.mm(x, self.h))
        return self.b + tr.mm(x, self.w) + tr.mm(tanh_x, self.u)
