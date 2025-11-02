import numpy as np

class CNN:
    def __init__(self, filter_size = (5,5), eps = 1e-9):
        self.filter_size = filter_size
        self.weight = np.zeros(filter_size) + eps
        



