import numpy as np


## Fault models
"""
    Fault(name=None)

The base class of fault models.
"""
class Fault:
    def __init__(self, name=None):
        self.name = name

    def __repr__(self):
        return f"Fault name: {self.name}"

    def get(t, u):
        raise NotImplementedError("`Fault` class needs `get`")

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)

"""
    LoE(time=0, index=1, level=1.0)

A fault class for loss of effectiveness (LoE).

# Parameters
time: from when LoE is applied
index: actuator index to which LoE is applied
level: effectiveness (e.g., level=1.0 means no fault)
"""
class LoE(Fault):
    def __init__(self, time=0, index=1, level=1.0, name="LoE"):
        super().__init__(name=name)
        self.time = time
        self.index = index
        self.level = level

    def __repr__(self):
        print(super().__repr__())
        return f"time = {self.time}, index = {self.index}, level = {self.level}"

    def get(self, t, u):
        if t >= self.time:
            u[self.index] *= self.level
        return u

"""
    Float(time=0, index=1)

A fault class for floating.

# Parameters
time: from when LoE is applied
index: actuator index to which LoE is applied
"""
class Float(LoE):
    def __init__(self, time=0, index=1, name="Float"):
        super().__init__(time=time, index=index, level=0.0, name=name)

    def get(self, t, u):
        return super().get(t, u)




if __name__ == "__main__":
    def test(fault):
        print(fault)
        n = 6
        ts = [0.0, 1.0, 2.0, 3.0]
        u = np.ones(n)
        for t in ts:
            print(f"Actuator command: {u}, time: {t}")
            u_fault = fault(t, u)
            print(f"Actual input: {u_fault}")
    # LoE
    faults = [
        LoE(time=2, index=1, level=0.2),
        Float(time=2, index=1),
    ]
    for fault in faults:
        test(fault)
