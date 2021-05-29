import numpy as np
from numpy import sin, cos
from math import sqrt

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import quat2angle


def sgn(s):
    if s > 0:
        return 1
    elif s < 0:
        return -1
    else:
        return 0


def sat(s, eps):
    if s > eps:
        return 1
    elif s < -eps:
        return -1
    else:
        return s/eps


def Herrera(s, delta):
    return s / (abs(s) + delta)


class SlidingModeController(BaseEnv):
    '''References:
    [1] Herrera, Marco, et al.
    "Sliding mode control: An approach to control a quadrotor."
    2015 Asia-Pacific Conference on Computer Aided System Engineering. IEEE, 2015.i
    [2] Runcharoon, Krittaya, and Varawan Srichatrapimuk.
    "Sliding mode control of quadrotor."
    2013 The International Conference on Technological Advances in Electrical, Electronics and Computer Engineering (TAEECE). IEEE, 2013.
    '''
    def __init__(self, env, ref):
        super().__init__()
        self.ref_ = np.vstack((ref[0:6], np.vstack(quat2angle(ref[6:10])[::-1]), ref[10:]))
        self.env = env

        self.d, self.m, self.g, self.J = env.plant.d, env.plant.m, env.plant.g, env.plant.J

    def get_action(self, obs, gammaTune, kdTune, dtype="sat"):
        # Tuning parameters
        gt_F, gt_M1, gt_M2, gt_M3 = gammaTune
        kt_F, kt_M1, kt_M2, kt_M3 = kdTune
        # model
        m, g = self.m, self.g
        Ix = self.J[0, 0]
        Iy = self.J[1, 1]
        Iz = self.J[2, 2]
        d = self.d
        # reference
        z_r, w_r = self.ref_[2], self.ref_[5]
        phi_r, theta_r, psi_r, p_r, q_r, r_r = self.ref_[6:]
        wd_r = 0
        pd_r = 0
        qd_r = 0
        rd_r = 0
        # observation
        obs = np.vstack((obs))
        obs_ = np.vstack((obs[0:6], np.vstack(quat2angle(obs[6:10])[::-1]), obs[10:]))
        z, w = obs_[2], obs_[5]
        phi, theta, psi, p, q, r = obs_[6:]
        # error term
        e_z = z_r - z
        e_z_dot = w_r - w
        e_phi = phi_r - phi
        e_phi_dot = p_r - p
        e_theta = theta_r - theta
        e_theta_dot = q_r - r
        e_psi = psi_r - psi
        e_psi_dot = r_r - r

        # sliding surface
        sF = e_z_dot + gt_F*e_z
        sM1 = e_phi_dot + gt_M1*e_phi
        sM2 = e_theta_dot + gt_M2*e_theta
        sM3 = e_psi_dot + gt_M3*e_psi

        # continuous part
        Feq = (g - gt_F*e_z_dot - wd_r) * m / (cos(phi)*cos(theta))
        M1eq = (gt_M1*e_phi_dot + pd_r - q*r*(Iy-Iz)/Ix)*Ix/d
        M2eq = (gt_M2*e_theta_dot + qd_r - p*r*(Iz-Ix)/Iy)*Iy/d
        M3eq = (gt_M3*e_psi_dot + rd_r - p*q*(Ix-Iy)/Iz)*Iz

        # discrete part
        if dtype == "sgn":
            Fd, M1d, M2d, M3d = sgn(sF), sgn(sM1), sgn(sM2), sgn(sM3)
        elif dtype == "sat":
            eps = 1
            Fd, M1d, M2d, M3d = sat(sF, eps), sat(sM1, eps), sat(sM2, eps), sat(sM3, eps)
        elif dtype == "Herrera":
            delta = 0.3
            Fd, M1d, M2d, M3d = Herrera(sF, delta), Herrera(sM1, delta), Herrera(sM2, delta), Herrera(sM3, delta)

        # F = Feq + kt_F*Fd
        F = Feq - kt_F*Fd
        M1 = M1eq + kt_M1*M1d
        M2 = M2eq + kt_M2*M2d
        M3 = M3eq + kt_M3*M3d

        action = np.vstack((F, M1, M2, M3))
        return action


if __name__ == "__main__":
    pass
