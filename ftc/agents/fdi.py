import numpy as np
from numpy import searchsorted as ss
from functools import reduce

from fym.core import BaseEnv, BaseSystem


def get_loe(actuator_faults, no_act):
    actuator_faults = sorted(actuator_faults, key=lambda x: x.time)
    loe = np.eye(no_act)
    yield loe
    for act_fault in actuator_faults:
        loe = loe.copy()
        loe[act_fault.index, act_fault.index] = act_fault.level
        yield loe


class SimpleFDI():
    def __init__(self, actuator_faults, no_act, delay=0., threshold=0.):
        self.delay = delay
        self.threshold = threshold

        self.loe = list(get_loe(actuator_faults, no_act))
        self.fault_times = np.array(
            [0] + sorted([x.time for x in actuator_faults]))

    def get(self, t):
        index = max(ss(self.fault_times, t - self.delay, side="right") - 1, 0)
        return self.loe[index]

    def get_true(self, t):
        index = ss(self.fault_times, t, side="right") - 1
        return self.loe[index]

    def get_index(self, t):
        return np.nonzero(np.diag(self.get(t)) < 1 - self.threshold)[0]


if __name__ == "__main__":
    from ftc.faults.actuator import LoE, LiP, Float

    actuator_faults = [
        LoE(time=3, index=0, level=0.7),
        LoE(time=6, index=2, level=0.8),
        LoE(time=1, index=0, level=0.6),
    ]
    fdi = SimpleFDI(actuator_faults, 6)
