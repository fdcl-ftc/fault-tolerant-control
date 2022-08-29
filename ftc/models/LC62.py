from copy import deepcopy
from functools import reduce

import fym
import numpy as np
from fym.utils.rot import angle2quat, dcm2quat, quat2angle, quat2dcm


def safeupdate(*configs):
    assert len(configs) > 1

    def _merge(base, new):
        assert isinstance(base, dict), f"{base} is not a dict"
        assert isinstance(new, dict), f"{new} is not a dict"
        out = deepcopy(base)
        for k, v in new.items():
            # assert k in out, f"{k} not in {base}"
            if isinstance(v, dict):
                if "grid_search" in v:
                    out[k] = v
                else:
                    out[k] = _merge(out[k], v)
            else:
                out[k] = v

        return out

    return reduce(_merge, configs)


class LC62(fym.BaseEnv):
    """LC62 Model
    Variables:
        pos: position in I-coord
        vel: velocity in I-coord
        quat: unit quaternion.
            Corresponding to the rotation matrix from I- to B-coord.
    """

    temp_y = 0.0
    # Aircraft paramters
    dx1 = 0.9325 + 0.049
    dx2 = 0.0725 - 0.049
    dx3 = 1.1725 - 0.049
    dy1 = 0.717 + temp_y
    dy2 = 0.717 + temp_y

    Ixx = 1.3 * 8.094
    Iyy = 1.3 * 9.125
    Izz = 1.3 * 16.8615
    Ixz = -1.3 * 0.308

    J = np.array([
        [Ixx, 0, Ixz],
        [0, Iyy, 0],
        [Ixz, 0, Izz],
    ])
    Jinv = np.linalg.inv(J)

    g = 9.81
    m = 41.97

    S1 = 0.2624
    S2 = 0.5898
    S = S1 + S2
    c = 0.551  # Main wing chord length
    b = 1.1  # Main wing half span
    d = 0.849  # Moment arm length

    inc = 0  # Wing incidence angle

    # Ele Pitch Moment Coefficient [1/rad]
    Cm_del_E = -0.676
    Cm_del_A = 0.001156

    # Rolling Moment Coefficient [1/rad]
    Cll_beta = -0.0518
    Cll_p = -0.4624
    Cll_r = 0.0218
    Cll_del_A = -0.0369 * 5
    Cll_del_R = 0.0026

    # Yawing Moment Coefficient [1/rad]
    Cn_beta = 0.0866
    Cn_p = -0.0048
    Cn_r = -0.0723
    Cn_del_A = -0.000385
    Cn_del_R = -0.0190

    # Y-axis Force Coefficient [1/rad]
    Cy_beta = -1.1269
    Cy_p = 0.0
    Cy_r = 0.2374
    Cy_del_R = 0.0534

    ENV_CONFIG = {
        "init": {
            "pos": np.zeros((3, 1)),
            "vel": np.zeros((3, 1)),
            "quat": np.vstack((1, 0, 0, 0)),
            "omega": np.zeros((3, 1)),
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__()
        self.pos = fym.BaseSystem(env_config["init"]["pos"])
        self.vel = fym.BaseSystem(env_config["init"]["vel"])
        self.quat = fym.BaseSystem(env_config["init"]["quat"])
        self.omega = fym.BaseSystem(env_config["init"]["omega"])

        self.e3 = np.vstack((0, 0, 1))

    def deriv(self, pos, vel, quat, omega, ctrls, vel_wind=np.zeros((3, 1)),
              omega_wind=np.zeros((3, 1))):

        pwms_rotor = ctrls[:6]
        pwms_pusher = ctrls[6:8]
        dels = ctrls[8:]  # control surfaces

        """ multicopter """
        FM_VTOL = self.B_VTOL(pwms_rotor, omega)

        """ fixed-wing """
        vel = vel - vel_wind
        omega = omega + omega_wind
        FM_Pusher = self.B_Pusher(pwms_pusher)
        FM_Fuselage = self.B_Fuselage(dels, vel, omega)
        # total force and moments
        FM = FM_VTOL + FM_Fuselage + FM_Pusher
        F, M = FM[0:3], FM[3:]

        """ disturbances """
        dv = np.zeros((3, 1))
        domega = self.Jinv @ np.zeros((3, 1))

        """ dynamics """
        dcm = quat2dcm(quat)
        dpos = dcm.T @ vel
        dvel = F / self.m - np.cross(omega, vel, axis=0) + dv
        p, q, r = np.ravel(omega)
        dquat = 0.5 * np.array(
            [[0.0, -p, -q, -r], [p, 0.0, r, -q], [q, -r, 0.0, p], [r, q, -p, 0.0]]
        ).dot(quat)
        eps = 1 - (quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2)
        k = 1
        dquat = dquat + k * eps * quat
        domega = (
            self.Jinv @ (M - np.cross(omega, self.J @ omega, axis=0)) + domega
        )
        return dpos, dvel, dquat, domega

    def set_dot(self, t, ctrls):
        """
        Parameters:
            controls: PWMs (rotor, pusher) and control surfaces
        """
        states = self.observe_list()
        dots = self.deriv(*states, ctrls)
        self.pos.dot, self.vel.dot, self.quat.dot, self.omega.dot = dots

    def B_VTOL(self, pwms_rotor, omega):
        """
        R1: mid right,   [CW]
        R2: mid left,    [CCW]
        R3: front left,  [CW]
        R4: rear right,  [CCW]
        R5: front right, [CCW]
        R6: rear left,   [CW]
        """
        rotors = (pwms_rotor - 1000) / 1000  # rotor thrust
        th = (- 19281*rotors**3 + 36503*rotors**2 - 992.75*rotors) * self.g / 1000
        tq = - 6.3961*rotors**3 + 12.092*rotors**2 - 0.3156*rotors
        Fx = Fy = 0
        Fz = - th[0] - th[1] - th[2] - th[3] - th[4] - th[5]
        l = self.dy1*(th[1] + th[2] + th[5]) - self.dy2*(th[0] + th[3] + th[4])
        m = self.dx1*(th[2] + th[4]) - self.dx2*(th[0] + th[1]) - self.dx3*(th[3] + th[5])
        n = - tq[0] + tq[1] - tq[2] + tq[3] + tq[4] - tq[5]
        # compensation
        l = l - 0.5 * omega[0]
        m = m - 2 * omega[1]
        n = n - 1 * omega[2]
        return np.vstack((Fx, Fy, Fz, l, m, n))

    def B_Pusher(self, pwms_pusher):
        th = self.thrust_pusher(pwms_pusher)
        tq = self.torque_pusher(pwms_pusher)
        Fx = th[0] + th[1]
        Fy = Fz = 0
        l = tq[0] - tq[1]
        m = 0
        n = 0
        return np.vstack((Fx, Fy, Fz, l, m, n))

    def B_Fuselage(self, css, vel, omega):
        rho = self.get_rho(100)
        p, q, r = omega.ravel()
        u, v, w = vel.ravel()
        VT = np.linalg.norm(vel)
        alp = np.arctan2(w, u)
        beta = np.arcsin(v / (VT + 1e-10))
        qbar = 0.5 * rho * VT**2
        Cy_p, Cy_r, Cy_beta = self.Cy_p, self.Cy_r, self.Cy_beta
        Cll_p, Cll_r, Cll_beta = self.Cll_p, self.Cll_r, self.Cll_beta,
        Cn_p, Cn_r, Cn_beta = self.Cn_p, self.Cn_r, self.Cn_beta,
        Cy_del_R = self.Cy_del_R
        Cll_del_R, Cll_del_A = self.Cll_del_R, self.Cll_del_A
        Cm_del_E, Cm_del_A = self.Cm_del_E, self.Cm_del_A
        Cn_del_R, Cn_del_A = self.Cn_del_R, self.Cn_del_A
        S, S2, b, c, d = self.S, self.S2, self.b, self.c, self.d
        dela, dele, delr = css
        CL, CD, CM = self.aero_coeff(alp)
        Fx = - qbar * S * CD
        Fy = qbar * S * (p*Cy_p + r*Cy_r + beta*Cy_beta + delr*Cy_del_R)
        Fz = - qbar * S * CL
        l = qbar * b * (S * (p*Cll_p + r*Cll_r + beta*Cll_beta + delr*Cll_del_R)
                        + S2 * dela * Cll_del_A)
        m = qbar * c * S * (CM + dele*Cm_del_E + dela*Cm_del_A)
        n = qbar * 0.06894022 * d * (p*Cn_p + r*Cn_r + beta*Cn_beta
                                     + delr*Cn_del_R + dela*Cn_del_A)
        return np.vstack((Fx, Fy, Fz, l, m, n))

    def get_rho(self, altitude):
        pressure = 101325 * (1 - 2.25569e-5 * altitude) ** 5.25616
        temperature = 288.14 - 0.00649 * altitude
        return pressure / (287 * temperature)

    def thrust_pusher(self, x):
        xt = np.array([1000, 1200, 1255, 1310, 1365, 1420, 1475, 1530, 1585, 1640, 1695, 1750])
        yt = np.array([0, 1.39, 4.22, 7.89, 12.36, 17.60, 23.19, 29.99, 39.09, 46.14, 52.67, 59.69])
        p = np.polyfit(xt, yt, deg=1)
        return p[0]*x + p[1]

    def torque_pusher(self, x):
        xt = np.array([1000, 1200, 1255, 1310, 1365, 1420, 1475, 1530, 1585, 1640, 1695, 1750])
        yt = np.array([0, 0.12, 0.35, 0.66, 1.04, 1.47, 1.93, 2.50, 3.25, 3.83, 4.35, 4.95])
        p = np.polyfit(xt, yt, deg=1)
        return p[0]*x + p[1]

    def aero_coeff(self, alp):
        at = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16])
        CLt = np.array([0.1931, 0.4075, 0.6112, 0.7939, 0.9270, 1.0775, 0.9577, 1.0497, 1.0635])
        CDt = np.array([0.0617, 0.0668, 0.0788, 0.0948, 0.1199, 0.1504, 0.2105, 0.2594, 0.3128])
        Cmt = np.array([0.0406, 0.0141, -0.0208, -0.0480, -0.2717, -0.4096,
                        -0.1448, -0.2067, -0.2548])
        p_CL = np.polyfit(at, CLt, deg=1)
        p_CD = np.polyfit(at, CDt, deg=1)
        p_Cm = np.polyfit(at, Cmt, deg=1)
        CL = p_CL[0]*alp + p_CL[1]
        CD = p_CD[0]*alp + p_CD[1]
        Cm = p_Cm[0]*alp + p_Cm[1]
        return np.vstack((CL, CD, Cm))


if __name__ == "__main__":
    system = LC62()
    system.set_dot(t=0, ctrls=np.zeros((11, 1)))
    print(repr(system))
