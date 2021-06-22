import numpy as np
from numpy import searchsorted as ss
from fym.core import BaseEnv, BaseSystem


class SimpleFDI():
    def __init__(self, actuator_faults, no_act, delay=0.):
        self.delay = delay
        self.loe = [
            np.eye(no_act),
            *map(lambda x: x.level * np.eye(no_act),
                 sorted(actuator_faults, key=lambda x: x.time))
        ]
        self.fault_times = np.array([0] + [x.time for x in actuator_faults])

    def get(self, t):
        index = max(ss(self.fault_times, t - self.delay, side="right") - 1, 0)
        return self.loe[index]

    def get_real(self, t):
        index = ss(self.fault_times, t, side="right") - 1
        return self.loe[index]
