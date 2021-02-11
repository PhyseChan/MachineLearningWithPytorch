import torch as tr
import numpy as np


class GenerateBinarydata():
    def __init__(self, digits=4, numbers=100):
        self.digits = digits
        self.max_digits = digits + 1
        self.numbers = numbers

    def toBinary(self, num):
        num_binary = list()
        while (num // 2 is not 0):
            num_binary.append(num % 2)
            num = num // 2
            if num is 1:
                num_binary.append(1)
        num_binary.reverse()
        num_binary = (self.max_digits - len(num_binary)) * [0] + num_binary
        return num_binary

    def __call__(self):
        data_matrix = []  ## shape: n*2
        for i in range(100):
            a = np.random.randint(0, 2 ** self.digits - 1)
            b = np.random.randint(0, 2 ** self.digits - 1)
            sum = a+b
            a = self.toBinary(a)
            b = self.toBinary(b)
            data_matrix.append([a,b,sum])
        return data_matrix
