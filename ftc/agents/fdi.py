import numpy as np
from fym.core import BaseEnv, BaseSystem


class SimpleFDI(BaseSystem):
    def __init__(self, no_act, tau):
        super().__init__(np.eye(no_act))
        self.tau = tau

    def get_true(self, u, uc):
        w = np.hstack([
            ui / uci if not np.isclose(uci, 0)
            else 1 if (np.isclose(ui, 0) and np.isclose(uci, 0))
            else 0
            for ui, uci in zip(u, uc)])
        return np.diag(w)

    def get_index(self, W):
        fault_index = np.where(np.diag(W) != 1)[0]
        return fault_index

    def set_dot(self, W):
        What = self.state
        self.dot = - 1 / self.tau * (What - W)
