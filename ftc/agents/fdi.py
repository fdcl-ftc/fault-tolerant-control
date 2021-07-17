import numpy as np
from numpy import searchsorted as ss
from functools import reduce

from fym.core import BaseEnv, BaseSystem

import ftc.config

cfg = ftc.config.load(__name__)


def get_loe(actuator_faults, no_act):
    actuator_faults = sorted(actuator_faults, key=lambda x: x.time)
    loe = np.eye(no_act)
    yield loe
    for act_fault in actuator_faults:
        loe = loe.copy()
        loe[act_fault.index, act_fault.index] = act_fault.level
        yield loe


class SimpleFDI():
    def __init__(self, actuator_faults, no_act):
        self.delay = cfg.delay
        self.threshold = cfg.threshold

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
