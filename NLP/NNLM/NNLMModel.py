
import numpy as np
import torch as tr



class NNLM(tr.nn.Module):
    def __init__(self, vocab_size, repr_size=20, num_words=5):
        self.c = tr.nn.Embedding(vocab_size, repr_size)
        self.h = tr.nn.Parameter(tr.randn(repr_size * num_words, vocab_size))
        self.u = tr.nn.Parameter(tr.randn(repr_size * num_words, vocab_size))
        self.w = tr.nn.Parameter(tr.randn(repr_size * num_words, vocab_size))
        self.b = tr.nn.Parameter(tr.rand(vocab_size))
        self.d = tr.nn.Parameter(tr.rand(vocab_size))
        self.tanh = tr.nn.Tanh()

    def forward(self, x):
        x = self.c(x)
        x = x.view(1, -1)  # (num_words*repr, 1)
        return self.b + tr.mm(x, self.w) + tr.mm(self.u, self.tanh(self.d + tr.mm(x, self.h)))
