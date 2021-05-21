import numpy as np
import matplotlib.pyplot as plt

from fym.utils.rot import quat2angle
from fym.agents import LQR

from ftc.models.multicopter import Multicopter
import fym.logging
from fym.core import BaseEnv, BaseSystem


class LQRController:
    def __init__(self, Jinv, m, g):
        self.Jinv = Jinv
        self.m, self.g = m, g
        self.trim_forces = np.vstack([self.m * self.g, 0, 0, 0])

        A = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, -self.g, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, self.g, 0, 0, 0, 0, 0],
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
                      [-1/self.m, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, self.Jinv[0, 0], 0, 0],
                      [0, 0, self.Jinv[1, 1], 0],
                      [0, 0, 0, self.Jinv[2, 2]]])
        Q = np.diag([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
        R = np.diag([1, 1, 1, 1])

        self.K, *_ = LQR.clqr(A, B, Q, R)

    def transform(self, y):
        return np.vstack((y[0:6], np.vstack(quat2angle(y[6:10])[::-1]), y[10:]))

    def get_forces(self, obs, ref):
        x = self.transform(obs)
        x_ref = self.transform(ref)
        forces = -self.K.dot(x - x_ref) + self.trim_forces
        return forces
