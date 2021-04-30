import numpy as np
from copy import deepcopy
import sys


class Fault:
    """
        Fault(name=None)

    The base class of fault models.
    """
    def __init__(self, time=0, index=0, name=None):
        self.time = time
        self.index = index
        self.name = name

    def __repr__(self):
        return f"Fault name: {self.name}" + "\n" + f"time = {self.time}, index = {self.index}"

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def get(t, u):
        raise NotImplementedError("`Fault` class needs `get`")


class LoE(Fault):
    """
        LoE(time=0, index=0, level=1.0)

    A fault class for loss of effectiveness (LoE).

    # Parameters
    time: from when LoE is applied
    index: actuator index to which LoE is applied
    level: effectiveness (e.g., level=1.0 means no fault)
    """
    def __init__(self, time=0, index=0, level=1.0, name="LoE"):
        super().__init__(time=time, index=index, name=name)
        self.level = level

    def __repr__(self):
        _str = super().__repr__()
        return _str + f", level = {self.level}"

    def get(self, t, u):
        effectiveness = np.ones_like(u)
        if t >= self.time:
            effectiveness[self.index] = self.level
        return u * effectiveness


class Float(LoE):
    """
        Float(time=0, index=0)

    A fault class for floating.

    # Parameters
    time: from when LoE is applied
    index: actuator index to which LoE is applied
    """
    def __init__(self, time=0, index=0, name="Float"):
        super().__init__(time=time, index=index, level=0.0, name=name)

    def get(self, t, u):
        return super().get(t, u)


class LiP(Fault):
    """
        LiP(time=0, index=0)

    A fault class for lock-in-place (LiP).

    # Parameters
    time: from when LoE is applied
    index: actuator index to which LoE is applied
    """
    def __init__(self, time=0, index=0, name="LiP"):
        super().__init__(time=time, index=index, name=name)
        self.memory = None

    def get(self, t, u):
        if self.memory is None or t > self.memory[0]:
            if t <= self.time:
                self.memory = (t, u[self.index], False)
            elif not self.memory[2]:
                # Linear interpolation
                dudt = (u[self.index] - self.memory[1]) / (t - self.memory[0])
                u0 = self.memory[1] + dudt * (self.time - self.memory[0])
                self.memory = (self.time, u0, True)

        u_fault = deepcopy(u)
        if t > self.time:
            u_fault[self.index] = self.memory[1]
        return u_fault


# """
#     HardOver(time=0, index=0)

# A fault class for hard-over.

# # Parameters
# time: from when LoE is applied
# index: actuator index to which LoE is applied
# limit: actuator limit
# rate: increasing (or decreasing) rate
# """
# class HardOver(Fault):
#     def __init__(self, time, index, limit=1.0, rate=0.0, name="HardOver"):
#         super().__init__(time=time, index=index, name=name)

#     def get(self, t, u):
#         raise ValueError("TODO")


if __name__ == "__main__":
    def test(fault):
        print(fault)
        n = 6
        ts = [0, 1.99, 2.0, 2.01, 10]
        for t in ts:
            u = t*np.ones(n)
            print(f"Actuator command: {u}, time: {t}")
            u_fault = fault(t, u)
            print(f"Actual input: {u_fault}")
    # LoE
    faults = [
        LoE(time=2, index=1, level=0.1),
        Float(time=2, index=1),
        LiP(time=2, index=1, dt=0.01),
    ]
    for fault in faults:
        test(fault)

    # small test
    def small_test():
        fault = LiP(2, 0, 0.1)
        print(fault.get(1.9, [3, 0]))  # 3
        print(fault.get(1.95, [4, 1]))  # 4
        print(fault.get(2.05, [5, 2]))  # 4.5
        print(fault.get(1.96, [6, 3]))  # 6
        print(fault.get(2.06, [7, 4]))  # 4.5
    small_test()
