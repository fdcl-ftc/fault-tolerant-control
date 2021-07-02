import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize


class Grouping():
    def __init__(self, B):
        self.B = B.copy()

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
        self.B = B.copy()

    def get(self, What, fault_index=()):
        """Notes
        `fault_index` should be 1d array, e.g., `fault_index = [1]`.
        """
        BB = self.B @ What
        BB[:, fault_index] = np.zeros((4, 1))
        return np.linalg.pinv(BB)


class ConstrainedCA():
    """Reference:
    [1] W. Durham, K. A. Bordignon, and R. Beck, “Aircraft Control Allocation,”
    Aircraft Control Allocation, 2016, doi: 10.1002/9781118827789.

    Method:
    solve_lp: Linear Programming
    solve_opt: SLSQP
    """
    def __init__(self, B):
        self.B = B.copy()
        self.n_rotor = len(B[0])
        self.u_prev = np.zeros((self.n_rotor,))

    def get_faulted_B(self, fault_index):
        _B = np.delete(self.B, fault_index, 1)
        return _B

    def solve_lp(self, fault_index, v, rotor_min, rotor_max):
        n = self.n_rotor - len(fault_index)
        c = np.ones((n,))
        A_ub = np.vstack((-np.eye(n), np.eye(n)))
        b_ub = np.hstack((-rotor_min*np.ones((n,)), rotor_max*np.ones((n,))))
        A_eq = self.get_faulted_B(fault_index)
        b_eq = v.reshape((len(v),))

        sol = linprog(c, A_ub, b_ub, A_eq, b_eq, method="simplex")
        _u = sol.x
        for i in range(len(fault_index)):
            _u = np.insert(_u, fault_index[i], 0)
        return np.vstack(_u)

    def solve_opt(self, fault_index, v, rotor_min, rotor_max):
        n = self.n_rotor - len(fault_index)
        self.u_prev = np.delete(self.u_prev, fault_index)
        bnds = np.hstack((rotor_min*np.ones((n, 1)), rotor_max*np.ones((n, 1))))
        A_eq = self.get_faulted_B(fault_index)
        b_eq = v.reshape((len(v),))
        cost = (lambda u: np.linalg.norm(u, np.inf)
                + 1e5*np.linalg.norm(b_eq - A_eq.dot(u), 1))

        opts = {"ftol": 1e-5, "maxiter": 1000}
        sol = minimize(cost, self.u_prev, method="SLSQP",
                       bounds=bnds, options=opts)
        _u = sol.x
        for i in range(len(fault_index)):
            _u = np.insert(_u, fault_index[i], 0)
        self.u_prev = _u
        return np.vstack(_u)


if __name__ == "__main__":
    pass
