import numpy as np


"""
    HexacopterFault

`self.loe_func` is a function of the effectiveness of each rotor w.r.t. time.
"""
class HexacopterFault:
    def __init__(self, func):
        self.effectiveness_func = func

    def get(self, t):
        _effectiveness = self.effectiveness_func(t)
        if self.is_valid(_effectiveness):
            effectiveness = _effectiveness
        else:
            raise ValueError("Invalid effectiveness")
        return effectiveness

    def is_valid(self, effectiveness):
        valid = True
        if len(effectiveness) != 6:
            valid = False
        elif not all(eff > 0 for eff in effectiveness):
            valid = False
        return valid


if __name__ == "__main__":
    effectiveness_default = np.ones(6)
    effectiveness_fault = np.array([0.5, 1.0, 1.0, 1.0, 1.0, 1.0])
    def effectiveness_func(t):
        if t < 1:
            effectiveness = effectiveness_default
        else:
            effectiveness = effectiveness_fault
        return effectiveness
    fault = HexacopterFault(effectiveness_func)
    t = 0.0
    if any(fault.get(t) != effectiveness_default):
        raise ValueError("Something went wrong")
    else:
        print("Test passed")
    t = 10.0
    if any(fault.get(t) != effectiveness_fault):
        raise ValueError("Something went wrong")
    else:
        print("Test passed")
