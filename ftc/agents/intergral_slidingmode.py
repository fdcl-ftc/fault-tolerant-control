import numpy as np
from numpy import cos

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import quat2angle


def sat(s, eps):
    if s > eps:
        return 1
    elif s < -eps:
        return -1
    else:
        return s/eps


class IntergralSMC(BaseEnv):
    def __init__(self, J, m, g, d):
        super().__init__()
        self.J = J
        self.m, self.g, self.d = m, g, d
        self.p1 = BaseSystem(0)
        self.p2 = BaseSystem(0)
        self.p3 = BaseSystem(0)
        self.p4 = BaseSystem(0)

    def get_ddot(self):
        obs = self.obs
        action = self.action
        J = self.J
        m, g, d = self.m, self.g, self.d
        Ixx = J[0, 0]
        Iyy = J[1, 1]
        Izz = J[2, 2]

        obs = np.vstack((obs))
        obs_ = np.vstack((obs[0:6], np.vstack(quat2angle(obs[6:10])[::-1]), obs[10:]))
        phi, theta, psi, phid, thetad, psid = obs_[6:]
        F, M1, M2, M3 = action

        zdd = g - cos(phi)*cos(theta)*F/m
        phidd = (Iyy-Izz)/Ixx*thetad*psid + M1/Ixx*d
        thetadd = (Izz-Ixx)/Iyy*phid*psid + M2/Iyy*d
        psidd = (Ixx-Iyy)/Izz*phid*thetad + M3/Izz*d

        return zdd, phidd, thetadd, psidd

    def deriv(self):
        K = self.K
        Kc = self.Kc

        K1, K2, K3, K4 = K
        k11, k12 = K1
        k21, k22 = K2
        k31, k32 = K3
        k41, k42 = K4

        kc1, kc2, kc3, kc4 = Kc

        dp1 = -self.e_zdd + k12*self.e_zd + k11*self.e_z
        dp2 = -self.e_phidd + k22*self.e_phid + k21*self.e_phi
        dp3 = -self.e_thetadd + k32*self.e_thetad + k31*self.e_theta
        dp4 = -self.e_psidd + k42*self.e_psid + k41*self.e_psi

        return dp1, dp2, dp3, dp4

    def set_dot(self):
        dots = self.deriv()
        self.p1.dot, self.p2.dot, self.p3.dot, self.p4.dot = dots

    def get_FM(self, obs, action, ref, K, Kc, PHI, t):
        self.obs = obs
        self.action = action
        self.ref = ref
        self.K = K
        self.Kc = Kc
        K1, K2, K3, K4 = K
        k11, k12 = K1
        k21, k22 = K2
        k31, k32 = K3
        k41, k42 = K4
        kc1, kc2, kc3, kc4 = Kc
        PHI1, PHI2, PHI3, PHI4 = PHI
        # model
        J = self.J
        Ixx = J[0, 0]
        Iyy = J[1, 1]
        Izz = J[2, 2]
        m, g, d = self.m, self.g, self.d
        # observation
        obs = np.vstack((obs))
        obs_ = np.vstack((obs[0:6], np.vstack(quat2angle(obs[6:10])[::-1]), obs[10:]))
        z, zd = obs_[2], obs_[5]
        phi, theta, psi, phid, thetad, psid = obs_[6:]
        zdd, phidd, thetadd, psidd = self.get_ddot(obs, action)
        if t == 0:
            z0, z0d = obs_[2], obs_[5]
            phi0, phi0d, theta0, theta0d, psi0, psi0d = obs_[6:]
        # reference
        ref_ = np.vstack((ref[0:6], np.vstack(quat2angle(ref[6:10])[::-1]), ref[10:]))
        z_r, zd_r = ref_[2], ref_[5]
        phi_r, theta_r, psi_r, phid_r, thetad_r, psid_r = ref_[6:]
        zdd_r = 0
        phidd_r = 0
        thetadd_r = 0
        psidd_r = 0
        if t == 0:
            z0_r, z0d_r = ref_[2], ref_[5]
            phi0_r, phi0d_r, theta0_r, theta0d_r, psi0_r, psi0d_r = ref_[6:]
        # error definition
        self.e_z = z - z_r
        self.e_zd = zd - zd_r
        self.e_zdd = zdd - zdd_r
        self.e_phi = phi - phi_r
        self.e_phid = phid - phid_r
        self.e_phidd = phidd - phidd_r
        self.e_theta = theta - theta_r
        self.e_thetad = thetad - thetad_r
        self.e_thetadd = thetadd - thetadd_r
        self.e_psi = psi - psi_r
        self.e_psid = psid - psid_r
        self.e_psidd = psidd - psidd_r
        # h**(-1) function definition
        h1 = -m/cos(phi)/cos(theta)
        h2 = Ixx/d
        h3 = Iyy/d
        h4 = Izz/d
        # sliding surface
        s1 = self.e_z + self.e_zd - (z0-z0_r) - (z0d - z0d_r) + self.p1.state
        s2 = self.e_phi + self.e_phid - (phi0-phi0_r) - (phi0d - phi0d_r) + self.p2.state
        s3 = self.e_theta + self.e_thetad - (theta0-theta0_r) - (theta0d - theta0d_r) + self.p3.state
        s4 = self.e_psi + self.e_psid - (psi0-psi0_r) - (psi0d - psi0d_r) + self.p4.state
        # get FM
        F = h1*(zdd_r - k12*self.e_zd - k11*self.e_z - g) - h1*kc1*sat(s1, PHI1)
        M1 = h2*(zdd_r - k22*self.e_zd - k21*self.e_z - g) - h2*kc2*sat(s2, PHI2)
        M2 = h3*(zdd_r - k32*self.e_zd - k31*self.e_z - g) - h3*kc3*sat(s3, PHI3)
        M3 = h4*(zdd_r - k42*self.e_zd - k41*self.e_z - g) - h4*kc4*sat(s4, PHI4)

        action = np.vstack((F, M1, M2, M3))

        return action


if __name__ == "__main__":
    pass
