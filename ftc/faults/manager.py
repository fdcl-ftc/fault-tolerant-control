"""Fault managing module.

This module provides fault manager classes
that schedule the timing of each fault in a list of predefined faults.

Example:
    Define a manager::

        fault_manager = LoEManager([
            LoE(time=3, index=0, level=0.),
            LoE(time=6, index=2, level=0.),
        ], no_act=6)

    Predefined FDI::

        fdi = fault_manager.fdi

    Get faulty inputs::

        rotors = fault_manager.get_faulty_input(rotors)
"""
import numpy as np
from numpy import searchsorted

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
        index = max(searchsorted(
            self.fault_times, t - self.delay, side="right") - 1, 0)
        return self.loe[index]

    def get_true(self, t):
        index = searchsorted(self.fault_times, t, side="right") - 1
        return self.loe[index]

    def get_index(self, t):
        return np.nonzero(np.diag(self.get(t)) < 1 - self.threshold)[0]


class LoEManager:
    def __init__(self, faults, no_act):
        self.loe = list(get_loe(faults, no_act))
        self.fault_times = np.array([0] + sorted([x.time for x in faults]))
        self.fdi = SimpleFDI(faults, no_act)

    def get_faulty_input(self, t, u):
        index = searchsorted(self.fault_times, t, side="right") - 1
        return self.loe[index] @ u
