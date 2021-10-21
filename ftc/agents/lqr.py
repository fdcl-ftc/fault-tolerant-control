import numpy as np
import matplotlib.pyplot as plt

from fym.utils.rot import quat2angle
from fym.agents import LQR

from ftc.models.multicopter import Multicopter
import fym.logging
from fym.core import BaseEnv, BaseSystem

import ftc.config

cfg = ftc.config.load(__name__)


class LQRController:
    def __init__(self, Jinv, m, g):
        Q = cfg.LQRGain.Q
        R = cfg.LQRGain.R
        self.Jinv = Jinv
        self.m, self.g = m, g
        self.trim_forces = np.vstack([self.m * self.g, 0, 0, 0])

        A = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        B = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [-1/m, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, Jinv[0, 0], 0, 0],
                      [0, 0, Jinv[1, 1], 0],
                      [0, 0, 0, Jinv[2, 2]]])

        self.K, *_ = LQR.clqr(A, B, Q, R)

    def transform(self, y):
        """
        y = pos, vel, quat, omega
        """
        if len(y) == 13:
            return np.vstack((y[0:6],
                              np.vstack(quat2angle(y[6:10])[::-1]), y[10:]))

    def get_FM(self, obs, ref):
        x = self.transform(obs)
        x_ref = self.transform(ref)
        forces = -self.K.dot(x - x_ref) + self.trim_forces
        return forces
