""" References
[1] M. Faessler, A. Franchi and D. Scaramuzza, "Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag for Accurate Tracking of High-Speed Trajectories," in IEEE Robotics and Automation Letters, vol. 3, no. 2, pp. 620-626, April 2018, doi: 10.1109/LRA.2017.2776353.
"""
import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
from numpy import cos, sin

from ftc.models.LC62 import LC62


class FlatController:
    def __init__(self, env):
        super().__init__()
        self.m = env.plant.m
        self.g = env.plant.g
        self.J = env.plant.J
        self.e3 = np.vstack((0, 0, 1))

        self.posd = env.posd
        self.posd_1dot = env.posd_dot
        self.posd_2dot = nd.Derivative(self.posd, n=2)
        self.posd_3dot = nd.Derivative(self.posd, n=3)
        self.posd_4dot = nd.Derivative(self.posd, n=4)

        self.psid = env.psid
        self.psid_1dot = nd.Derivative(self.psid, n=1)
        self.psid_2dot = nd.Derivative(self.psid, n=2)

    def get_control(self, t):
        posd = self.posd(t)
        veld = self.posd_1dot(t)
        accd = self.posd_2dot(t)
        jerkd = self.posd_3dot(t)
        snapd = self.posd_4dot(t)

        psid = self.psid(t)
        psid_1dot = self.psid_1dot(t)
        psid_2dot = self.psid_2dot(t)

        Xc = np.vstack((cos(psid), sin(psid), 0))
        Yc = np.vstack((-sin(psid), cos(psid), 0))

        a = self.g * self.e3 - accd

        # Rotation matrix
        Xb = np.cross(Yc.T, a.T).T / np.linalg.norm(np.cross(Yc.T, a.T))
        Yb = np.cross(a.T, Xb.T).T / np.linalg.norm(np.cross(a.T, Xb.T))
        Zb = np.cross(Xb.T, Yb.T).T
        R = np.hstack((Xb, Yb, Zb))

        # Total thrust
        c = np.dot(Zb.T, a)
        cdot = -np.dot(Zb.T, jerkd)
        Fz = self.m * c

        # Angular rates
        p = np.dot(Yb.T, jerkd) / c
        q = -np.dot(Xb.T, jerkd) / c
        r = (psid_1dot * Xc.T @ Xb + q * Yc.T @ Zb) / np.linalg.norm(
            np.cross(Yc.T, Zb.T)
        )
        omega = np.vstack((p, q, r))

        # Derivatives of angular rates
        pdot = Yb.T @ snapd / c - 2 * cdot * (Yb.T @ jerkd / c**2) + q * r
        qdot = -Xb.T @ snapd / c + 2 * cdot * (Xb.T @ jerkd / c**2) - p * r
        rdot = (
            psid_2dot * Xc.T @ Xb
            + 2 * psid_1dot * (r * Xc.T @ Yb - q * Xc.T @ Zb)
            - p * q * Yc.T @ Yb
            - p * r * Yc.T @ Zb
            + qdot * Yc.T @ Zb
        ) / np.linalg.norm(np.cross(Yc.T, Zb.T))
        omega_dot = np.vstack((pdot, qdot, rdot))

        M = self.J @ omega_dot + np.cross(omega.T, (self.J @ omega).T).T

        FM = np.vstack((0, 0, Fz, M))
        return FM


if __name__ == "__main__":

    class Env:
        def __init__(self):
            super().__init__()
            self.plant = LC62()

        def posd(self, t):
            posd = np.vstack((np.sin(t), np.cos(t), -t))
            return posd

        def psid(self, t):
            return 0

    env = Env()
    tspan = np.linspace(0, 20, 200)
    ctrl = FlatController(env)
    FM_traj = np.empty((6, 0))
    for t in tspan:
        FM = ctrl.get_control(t)
        FM_traj = np.append(FM_traj, FM, axis=1)

    """Figure - Forces & Moments trajectories"""
    fig, axes = plt.subplots(3, 2, squeeze=False, sharex=True)

    """ Column 1 - Forces """
    ax = axes[0, 0]
    ax.plot(tspan, FM_traj[0, :], "k-")
    ax.set_ylabel(r"$F_x$")
    ax.legend(["Response"], loc="upper right")

    ax = axes[1, 0]
    ax.plot(tspan, FM_traj[1, :], "k-")
    ax.set_ylabel(r"$F_y$")

    ax = axes[2, 0]
    ax.plot(tspan, FM_traj[2, :], "k-")
    ax.set_ylabel(r"$F_z$")
    ax.set_xlabel("Time, sec")

    """ Column 2 - Moments """
    ax = axes[0, 1]
    ax.plot(tspan, FM_traj[3, :], "k-")
    ax.set_ylabel(r"$M_x$")

    ax = axes[1, 1]
    ax.plot(tspan, FM_traj[4, :], "k-")
    ax.set_ylabel(r"$M_y$")

    ax = axes[2, 1]
    ax.plot(tspan, FM_traj[5, :], "k-")
    ax.set_ylabel(r"$M_z$")
    ax.set_xlabel("Time, sec")

    plt.tight_layout()

    plt.show()
