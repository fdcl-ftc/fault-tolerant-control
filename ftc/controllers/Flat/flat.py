""" References
[1] M. Faessler, A. Franchi and D. Scaramuzza, "Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag for Accurate Tracking of High-Speed Trajectories," in IEEE Robotics and Automation Letters, vol. 3, no. 2, pp. 620-626, April 2018, doi: 10.1109/LRA.2017.2776353.
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin


class FlatController:
    def __init__(self, m, g, J):
        super().__init__()
        self.m = m
        self.g = g * np.vstack((0, 0, 1))
        self.J = J

    def get_ref(self, t, *args):
        # desired trajectories
        posd = np.vstack([np.sin(t), np.cos(t), -t])
        posd_1dot = np.vstack([np.cos(t), -np.sin(t), -1])
        posd_2dot = np.vstack([-np.sin(t), -np.cos(t), 0])
        posd_3dot = np.vstack([-np.cos(t), np.sin(t), 0])
        posd_4dot = np.stack([np.sin(t), np.cos(t), 0])
        refs = {
            "posd": posd,
            "posd_1dot": posd_1dot,
            "posd_2dot": posd_2dot,
            "posd_3dot": posd_3dot,
            "posd_4dot": posd_4dot,
        }
        return [refs[key] for key in args]

    def get_control(self, t):
        posd, veld, accd, jerkd, snapd = self.get_ref(
            t, "posd", "posd_1dot", "posd_2dot", "posd_3dot", "posd_4dot"
        )
        psid = 0
        psid_1dot = 0
        psid_2dot = 0

        Xc = np.vstack((cos(psid), sin(psid), 0))
        Yc = np.vstack((-sin(psid), cos(psid), 0))

        a = self.g - accd

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
    m, g = 1, 9.81
    J = np.diag([1, 1, 1])

    tspan = np.linspace(0, 20, 200)
    ctrl = FlatController(m, g, J)
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
