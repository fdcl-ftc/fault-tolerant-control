import numpy as np


class Grouping():
    def __init__(self, B):
        self.B = B

    def get(self, fault_index):
        if fault_index in [0, 1]:
            self.B[:, :2] = np.zeros((4, 2))
            G = self.B

        elif fault_index in [2, 3]:
            self.B[:, 2:4] = np.zeros((4, 2))
            G = self.B

        elif fault_index in [4, 5]:
            self.B[:, 4:] = np.zeros((4, 2))
            G = self.B

        return G


class CA():
    def __init__(self, B):
        self.B = B

    def get(self, fault_index):
        self.B[:, fault_index] = np.zeros((4, 1))
        BB = self.B

        return BB


if __name__ == "__main__":
    pass
