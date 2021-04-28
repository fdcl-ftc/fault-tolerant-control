import numpy as np
from copy import deepcopy
import sys


## Fault models
"""
    Fault(name=None)

The base class of fault models.
"""
class Fault:
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


"""
    LoE(time=0, index=0, level=1.0)

A fault class for loss of effectiveness (LoE).

# Parameters
time: from when LoE is applied
index: actuator index to which LoE is applied
level: effectiveness (e.g., level=1.0 means no fault)
"""
class LoE(Fault):
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


"""
    Float(time=0, index=0)

A fault class for floating.

# Parameters
time: from when LoE is applied
index: actuator index to which LoE is applied
"""
class Float(LoE):
    def __init__(self, time=0, index=0, name="Float"):
        super().__init__(time=time, index=index, level=0.0, name=name)

    def get(self, t, u):
        return super().get(t, u)


"""
    LiP(time=0, index=0)

A fault class for lock-in-place (LiP).

# Parameters
time: from when LoE is applied
index: actuator index to which LoE is applied
"""
class LiP(Fault):
    def __init__(self, time=0, index=0, dt=0.00, name="LiP"):
        if dt < 0 or dt < sys.float_info.epsilon:  # negative or too small
            raise ValueError("Invalid time step to detect actuator value at the fault time")
        super().__init__(time=time, index=index, name=name)
        self.dt = dt
        self.u_locked = None  # by default

    def __repr__(self):
        _str = super().__repr__()
        return _str + f", dt = {self.dt}"

    def get(self, t, u):
        if abs(t - self.time) < 0.5*self.dt:  # consider errors e.g. machine precision
            self.u_locked = u
        if self.u_locked is None:
            return u
        else:
            return self.u_locked


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
