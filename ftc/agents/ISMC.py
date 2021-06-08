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


class IntegralSMC(BaseEnv):
    '''
    reference
    Ban Wang, Youmin Zhang, An Adaptive Fault-Tolerant Sliding Mode Control
    Allocation Scheme for Multirotor Helicopter Subjected to
    Actuator Faults, IEEE Transactions on industrial electronics, Vol. 65,
    No. 5, May 2018
    '''

    def __init__(self, J, m, g, d, ic, ref0):
        super().__init__()
        self.ic_ = np.vstack((ic[0:6], np.vstack(quat2angle(ic[6:10])[::-1]), ic[10:]))
        self.ref0_ = np.vstack((ref0[0:6], np.vstack(quat2angle(ref0[6:10])[::-1]), ref0[10:]))
        self.J, self.m, self.g, self.d = J, m, g, d
        # error integral of x, y, z, roll, pitch, yaw
        self.P = BaseSystem(np.vstack((self.ic_[0] - self.ref0_[0],
                                       self.ic_[1] - self.ref0_[1],
                                       self.ic_[2] - self.ref0_[2],
                                       self.ic_[6] - self.ref0_[6],
                                       self.ic_[7] - self.ref0_[7],
                                       self.ic_[8] - self.ref0_[8])))

    def deriv(self, obs, ref):
        # observation
        obs = np.vstack((obs))
        obs_ = np.vstack((obs[0:6], np.vstack(quat2angle(obs[6:10])[::-1]), obs[10:]))
        x, y, z = obs_[0:3]
        phi, theta, psi = obs_[6:9]
        # reference
        ref = np.vstack((ref))
        ref_ = np.vstack((ref[0:6], np.vstack(quat2angle(ref[6:10])[::-1]), ref[10:]))
        x_r, y_r, z_r = ref_[0:3]
        phi_r, theta_r, psi_r = ref_[6:9]
        dP = np.vstack((x - x_r,
                        y - y_r,
                        z - z_r,
                        phi - phi_r,
                        theta - theta_r,
                        psi - psi_r))

        return dP

    def set_dot(self, obs, ref):
        dot = self.deriv(obs, ref)
        self.P.dot = dot

    def get_FM(self, obs, ref, p, K, Kc, PHI, t):
        p = np.vstack((p))
        px, py, pz, pphi, ptheta, ppsi = p
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
        x, y, z, xd, yd, zd = obs_[0:6]
        phi, theta, psi, phid, thetad, psid = obs_[6:]
        # reference
        ref_ = np.vstack((ref[0:6], np.vstack(quat2angle(ref[6:10])[::-1]), ref[10:]))
        x_r, y_r, z_r, xd_r, yd_r, zd_r = ref_[0:6]
        phi_r, theta_r, psi_r, phid_r, thetad_r, psid_r = ref_[6:]
        zdd_r = 0
        phidd_r = 0
        thetadd_r = 0
        psidd_r = 0
        # initial condition
        z0, z0d = self.ic_[2], self.ic_[5]
        phi0, theta0, psi0, phi0d, theta0d, psi0d = self.ic_[6:]
        z0_r, z0d_r = self.ref0_[2], self.ref0_[5]
        phi0_r, theta0_r, psi0_r, phi0d_r, theta0d_r, psi0d_r = self.ref0_[6:]
        # PD control for position tracking (get phi_ref, theta_ref)
        e_x = x - x_r
        e_xd = xd - xd_r
        e_y = y - y_r
        e_yd = yd - yd_r
        kp1, kd1, ki1 = np.array([0.25, 0.3, 0.1])
        kp2, kd2, ki2 = np.array([0.25, 0.3, 0.1])
        phi_r = -(kp1*e_y + kd1*e_yd + ki1*py)
        theta_r = kp2*e_x + kd2*e_xd + ki2*px
        # error definition
        e_z = z - z_r
        e_zd = zd - zd_r
        e_phi = phi - phi_r
        e_phid = phid - phid_r
        e_theta = theta - theta_r
        e_thetad = thetad - thetad_r
        e_psi = psi - psi_r
        e_psid = psid - psid_r
        # h**(-1) function definition
        h1 = -m/cos(phi)/cos(theta)
        h2 = Ixx/d
        h3 = Iyy/d
        h4 = Izz
        # sliding surface
        s1 = e_zd + k12*e_z + k11*pz - k12*(z0-z0_r) - (z0d-z0d_r)
        s2 = e_phid + k22*e_phi + k21*pphi - k22*(phi0-phi0_r) - (phi0d-phi0d_r)
        s3 = e_thetad + k32*e_theta + k31*ptheta - k32*(theta0-theta0_r) - (theta0d-theta0d_r)
        s4 = e_psid + k42*e_psi + k41*ppsi - k42*(psi0-psi0_r) - (psi0d-psi0d_r)
        # get FM
        F = h1*(zdd_r - k12*e_zd - k11*e_z - g) - h1*kc1*sat(s1, PHI1)
        M1 = h2*(phidd_r - k22*e_phid - k21*e_phi - (Iyy-Izz)/Ixx*thetad*psid) - h2*kc2*sat(s2, PHI2)
        M2 = h3*(thetadd_r - k32*e_thetad - k31*e_theta - (Izz-Ixx)/Iyy*phid*psid) - h3*kc3*sat(s3, PHI3)
        M3 = h4*(psidd_r - k42*e_psid - k41*e_psi - (Ixx-Iyy)/Izz*phid*thetad) - h4*kc4*sat(s4, PHI4)

        action = np.vstack((F, M1, M2, M3))

        return action


if __name__ == "__main__":
    pass
