import numpy as np
from scipy.optimize import linprog


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
        """Notes
        `fault_index` should be 1d array, e.g., `fault_index = [1]`.
        """
        self.B[:, fault_index] = np.zeros((4, 1))
        BB = self.B

        return BB


class CCA():
    def __init__(self, B):
        self.B = B
        self.n_rotor = len(B[0])

    def get_faulted_B(self, fault_index):
        _B = np.delete(self.B, fault_index[0], 1)

        return _B

    def solve_lp(self, fault_index, v, rotor_max, rotor_min):
        n = self.n_rotor - len(fault_index)
        c = np.ones((n,))
        A_ub = np.vstack((np.eye(n), -np.eye(n)))
        b_ub = np.hstack((rotor_max*np.ones((n,)), -rotor_min*np.ones((n,))))
        A_eq = self.get_faulted_B(fault_index)
        b_eq = v.reshape((len(v),))

        sol = linprog(c, A_ub, b_ub, A_eq, b_eq, method="interior-point")
        _u = sol.x

        return np.vstack(np.insert(_u, fault_index[0], 0))


if __name__ == "__main__":
    pass
