""" References
[1] M. Faessler, A. Franchi and D. Scaramuzza, "Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag for Accurate Tracking of High-Speed Trajectories," in IEEE Robotics and Automation Letters, vol. 3, no. 2, pp. 620-626, April 2018, doi: 10.1109/LRA.2017.2776353.
"""

import fym
import numpy as np
from fym.utils.rot import quat2angle, quat2dcm
from numpy import cos, sin, tan

class FlatController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        self.m = env.plant.m
        self.g = env.plant.g * np.vstack((0,0,1))
        self.J = env.plant.J


    def get_control(self, t, env):
        # pos, vel, quat, omega = env.plant.observe_list()
        # ang = np.vstack(quat2angle(quat)[::-1])

        posd, veld, accd, jerkd, snapd = env.get_ref(t, "posd", "posd_1dot", "posd_2dot", "posd_3dot", "posd_4dot")
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

        # angular rates
        p = np.dot(Yb.T, jerkd) / c
        q = -np.dot(Xb.T, jerkd) / c
        r = (psid_1dot * Xc.T @ Xb + q * Yc.T @ Zb) / np.linalg.norm(np.cross(Yc.T, Zb.T))
        omega = np.vstack((p, q, r))

        # derivatives of angular rates
        pdot = Yb.T @ snapd/c - 2 * cdot * (Yb.T @ jerkd/c**2) + q*r
        qdot = -Xb.T @ snapd/c + 2 * cdot * (Xb.T @ jerkd/c**2) - p*r
        rdot = (psid_2dot * Xc.T @ Xb + 2 * psid_1dot * (r*Xc.T@Yb - q*Xc.T@Zb) - p*q*Yc.T@Yb - p*r*Yc.T@Zb + qdot*Yc.T@Zb) / np.linalg.norm(np.cross(Yc.T, Zb.T))
        omega_dot = np.vstack((pdot, qdot, rdot))

        M = self.J @ omega_dot + np.cross(omega.T, (self.J @ omega).T).T

        FM_traj = np.vstack((0,0,Fz, M))
        controller_info = {
            "posd": posd,
            "veld": veld,
            "psid": psid,
            "omegad": omega,
        }
        return FM_traj, controller_info
        













        


