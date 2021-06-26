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
    count = 0
    num = 0

    def __init__(self, time=0, index=0, level=1.0, name="LoE"):
        super().__init__(time=time, index=index, name=name)
        self.level = level
        LoE.num += 1

    def __repr__(self):
        _str = super().__repr__()
        return _str + f", level = {self.level}"

    def get(self, t, u):
        if LoE.count == 0:
            LoE.u = u
            effectiveness = np.ones_like(u)
            last_time = np.zeros_like(u)
        else:
            effectiveness, last_time = u
        LoE.count += 1
        if t > self.time and self.time > last_time[self.index]:
            last_time[self.index] = self.time
            effectiveness[self.index] = self.level
        if LoE.count == LoE.num:
            LoE.count = 0
            return LoE.u * effectiveness
        else:
            return effectiveness, last_time


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
    import matplotlib.pyplot as plt
    import fym

    faults = [
        LoE(time=2, index=0, level=0.8),
        LoE(time=10, index=0, level=0.2),
        LoE(time=6, index=0, level=0.5),
        LoE(time=8, index=2, level=0.4),
        LoE(time=4, index=1, level=0.7),
    ]

    logger = fym.Logger("data.h5")
    clock = fym.core.Clock(dt=0.01, max_t=12)
    rotors_cmd = np.ones((6, 1))
    for t in clock.tspan:
        rotors = rotors_cmd
        for fault in faults:
            rotors = fault(t, rotors)
        logger.record(t=t, rotors_cmd=rotors_cmd, rotors=rotors)

    logger.close()

    data = fym.parser.parse(fym.load("data.h5"))
    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        plt.axes(ax)
        plt.plot(data.t, data.rotors[:, i, 0], "k-")
        plt.plot(data.t, data.rotors_cmd[:, i, 0], "r-.")
        ax.set(ylabel=f"Rotor {i}", ylim=(-0.1, 1.1))
        if ax.get_subplotspec().is_last_row():
            ax.set(xlabel="Time, sec")
    plt.tight_layout()
    plt.show()
