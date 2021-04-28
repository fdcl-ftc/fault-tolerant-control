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
    def __init__(self, time=0, index=1, level=1.0, name = "LoE"):
        super().__init__(name=name)
        self.time = time
        self.index = index
        self.level = level

    def __repr__(self):
        print(super().__repr__())
        return f"time = {self.time}, index = {self.index}, level = {self.level}"

    def get(self, t, u):
        effectiveness = np.ones_like(u)
        if t >= self.time:
            effectiveness[self.index] = self.level
        return u * effectiveness

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


## Deprecated
# """
#     HexacopterFault

# `self.loe_func` is a function of the effectiveness of each rotor w.r.t. time.
# """
# class HexacopterFault:
#     def __init__(self, func):
#         self.effectiveness_func = func

#     def get(self, t):
#         _effectiveness = self.effectiveness_func(t)
#         if self.is_valid(_effectiveness):
#             effectiveness = _effectiveness
#         else:
#             raise ValueError("Invalid effectiveness")
#         return effectiveness

#     def is_valid(self, effectiveness):
#         valid = True
#         if effectiveness.shape != (6, 6):
#             valid = False
#         elif not all(eff >= 0.0 for eff in effectiveness.flatten()):
#             valid = False
#         return valid


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

    ## Deprecated
    # effectiveness_default = np.diag(np.ones(6))
    # effectiveness_fault = np.diag(np.array([0.5, 1.0, 1.0, 1.0, 1.0, 1.0]))
    # def effectiveness_func(t):
    #     if t < 1:
    #         effectiveness = effectiveness_default
    #     else:
    #         effectiveness = effectiveness_fault
    #     return effectiveness
    # fault = HexacopterFault(effectiveness_func)
    # t = 0.0
    # if any(fault.get(t).flatten() != effectiveness_default.flatten()):
    #     raise ValueError("Something went wrong")
    # else:
    #     print("Test passed")
    # t = 10.0
    # if any(fault.get(t).flatten() != effectiveness_fault.flatten()):
    #     raise ValueError("Something went wrong")
    # else:
    #     print("Test passed")
