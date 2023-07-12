import fym
import numpy as np
import scipy
from fym.utils.rot import angle2quat, dcm2quat, quat2angle, quat2dcm
from numpy import cos, sin

from ftc.utils import safeupdate


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
    dx1 = 0.9325 + 0.049  # dx1 = 0.9815
    dx2 = 0.0725 - 0.049  # dx2 = 0.0235
    dx3 = 1.1725 - 0.049  # dx3 = 1.1235
    dy1 = 0.717 + temp_y
    dy2 = 0.717 + temp_y

    Ixx = 1.3 * 8.094  # [kg * m^2]
    Iyy = 1.3 * 9.125  # [kg * m^2]
    Izz = 1.3 * 16.8615  # [kg * m^2]
    Ixz = -1.3 * 0.308  # [kg * m^2]

    J = np.array(
        [
            [Ixx, 0, Ixz],
            [0, Iyy, 0],
            [Ixz, 0, Izz],
        ]
    )
    Jinv = np.linalg.inv(J)

    g = 9.81  # [m / sec^2]
    m = 41.97  # [kg]

    S1 = 0.2624  # [m^2]
    S2 = 0.5898  # [m^2]
    S = S1 + S2
    St = 0.06894022  # [m^2]
    c = 0.551  # Main wing chord length [m]
    b = 1.1  # Main wing half span [m]
    d = 0.849  # Moment arm length [m]

    inc = 0  # Wing incidence angle

    # Ele Pitch Moment Coefficient [1 / rad]
    Cm_del_A = 0.001156
    Cm_del_E = -0.676

    # Rolling Moment Coefficient [1 / rad]
    Cll_beta = -0.0518
    Cll_p = -0.4624
    Cll_r = 0.0218
    Cll_del_A = -0.0369 * 5
    Cll_del_R = 0.0026

    # Yawing Moment Coefficient [1 / rad]
    Cn_beta = 0.0866
    Cn_p = -0.0048
    Cn_r = -0.0723
    Cn_del_A = -0.000385
    Cn_del_R = -0.0190

    # Y-axis Force Coefficient [1 / rad]
    Cy_beta = -1.1269
    Cy_p = 0.0
    Cy_r = 0.2374
    Cy_del_R = 0.0534

    """
    pwm: 1000 ~ 2000
    th_p: pusher thrust [N]
    tq_p: pusher torque [N * m]
    """
    tables = {
        "alp": np.deg2rad(np.array([0, 2, 4, 6, 8, 10, 12, 14, 16])),  # [rad]
        "CL": np.array(
            [0.1931, 0.4075, 0.6112, 0.7939, 0.9270, 1.0775, 0.9577, 1.0497, 1.0635]
        ),
        "CD": np.array(
            [0.0617, 0.0668, 0.0788, 0.0948, 0.1199, 0.1504, 0.2105, 0.2594, 0.3128]
        ),
        "Cm": np.array(
            [
                0.0406,
                0.0141,
                -0.0208,
                -0.0480,
                -0.2717,
                -0.4096,
                -0.1448,
                -0.2067,
                -0.2548,
            ]
        ),
        "pwm": np.array(
            [1000, 1200, 1255, 1310, 1365, 1420, 1475, 1530, 1585, 1640, 1695, 1750]
        ),
        "th_p": np.array(
            [
                0,
                1.39,
                4.22,
                7.89,
                12.36,
                17.60,
                23.19,
                29.99,
                39.09,
                46.14,
                52.67,
                59.69,
            ]
        ),
        "tq_p": np.array(
            [0, 0.12, 0.35, 0.66, 1.04, 1.47, 1.93, 2.50, 3.25, 3.83, 4.35, 4.95]
        ),
        "th_r": [-19281, 36503, -992.75, 0],
        "tq_r": [-6.3961, 12.092, -0.3156, 0],
    }

    control_limits = {
        "pwm": (1000, 2000),
        "dela": np.deg2rad((-10, 10)),
        "dele": np.deg2rad((-10, 10)),
        "delr": np.deg2rad((-10, 10)),
    }

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
        # if VT >=40, the initial guess, z0, should be changed from default to 2000
        self.x_trims, self.u_trims_fixed = self.get_trim_fixed(fixed={"h": 10, "VT": 0})
        self.u_trims_vtol = self.get_trim_vtol(
            fixed={"x_trims": self.x_trims, "u_trims_fixed": self.u_trims_fixed}
        )

    def deriv(self, pos, vel, quat, omega, FM):
        F, M = FM[0:3], FM[3:]
        dcm = quat2dcm(quat)

        """ disturbances """
        dv = np.zeros((3, 1))
        domega = self.Jinv @ np.zeros((3, 1))

        """ dynamics """
        dpos = dcm.T @ vel
        dvel = F / self.m - np.cross(omega, vel, axis=0) + dv
        p, q, r = np.ravel(omega)
        dquat = 0.5 * np.array(
            [[0.0, -p, -q, -r], [p, 0.0, r, -q], [q, -r, 0.0, p], [r, q, -p, 0.0]]
        ).dot(quat)
        eps = 1 - (quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2)
        k = 1
        dquat = dquat + k * eps * quat
        domega = self.Jinv @ (M - np.cross(omega, self.J @ omega, axis=0)) + domega
        return dpos, dvel, dquat, domega

    def set_dot(self, t, FM):
        states = self.observe_list()
        dots = self.deriv(*states, FM)
        self.pos.dot, self.vel.dot, self.quat.dot, self.omega.dot = dots

    def get_FM(
        self,
        pos,
        vel,
        quat,
        omega,
        ctrls,
        vel_wind=np.zeros((3, 1)),
        omega_wind=np.zeros((3, 1)),
    ):
        """
        ctrls: PWMs (rotor, pusher) and control surfaces
        """
        pwms_rotor = ctrls[:6]
        pwms_pusher = ctrls[6:8]
        dels = ctrls[8:]  # control surfaces

        """ multicopter """
        FM_VTOL = self.B_VTOL(pwms_rotor, omega)

        """ fixed-wing """
        FM_Pusher = self.B_Pusher(pwms_pusher)
        FM_Fuselage = self.B_Fuselage(dels, pos, vel - vel_wind, omega + omega_wind)
        FM_Gravity = self.B_Gravity(quat)

        # total force and moments
        FM = FM_VTOL + FM_Fuselage + FM_Pusher + FM_Gravity
        return FM

    def B_VTOL(self, pwms_rotor, omega):
        """
        R1: mid right,   [CW]
        R2: mid left,    [CCW]
        R3: front left,  [CW]
        R4: rear right,  [CCW]
        R5: front right, [CCW]
        R6: rear left,   [CW]
        """
        rcmds = self.pwm2cmd(pwms_rotor)
        th = (-19281 * rcmds**3 + 36503 * rcmds**2 - 992.75 * rcmds) * self.g / 1000
        tq = -6.3961 * rcmds**3 + 12.092 * rcmds**2 - 0.3156 * rcmds
        # th = np.polyval(self.tables["th_r"], rcmds) * self.g / 1000
        # tq = np.polyval(self.tables["tq_r"], rcmds)
        Fx = Fy = 0
        Fz = -th[0] - th[1] - th[2] - th[3] - th[4] - th[5]
        l = self.dy1 * (th[1] + th[2] + th[5]) - self.dy2 * (th[0] + th[3] + th[4])
        m = (
            self.dx1 * (th[2] + th[4])
            - self.dx2 * (th[0] + th[1])
            - self.dx3 * (th[3] + th[5])
        )
        n = -tq[0] + tq[1] - tq[2] + tq[3] + tq[4] - tq[5]
        # compensation
        l = l - 0.5 * np.rad2deg(omega[0])
        m = m - 2 * np.rad2deg(omega[1])
        n = n - 1 * np.rad2deg(omega[2])
        return np.vstack((Fx, Fy, Fz, l, m, n))

    def B_Pusher(self, pwms_pusher):
        th = np.interp(pwms_pusher, self.tables["pwm"], self.tables["th_p"])
        tq = np.interp(pwms_pusher, self.tables["pwm"], self.tables["tq_p"])
        Fx = th[0] + th[1]
        Fy = Fz = 0
        l = tq[0] - tq[1]
        m = n = 0
        return np.vstack((Fx, Fy, Fz, l, m, n))

    def B_Fuselage(self, dels, pos, vel, omega):
        rho = self.get_rho(-pos[2])
        u, v, w = np.ravel(vel)
        p, q, r = np.ravel(omega)
        VT = np.linalg.norm(vel)
        alp = np.arctan2(w, u)
        if np.isclose(VT, 0):
            beta = 0
        else:
            beta = np.arcsin(v / VT)
        qbar = 0.5 * rho * VT**2
        Cy_p, Cy_r, Cy_beta = self.Cy_p, self.Cy_r, self.Cy_beta
        Cll_p, Cll_r, Cll_beta = (
            self.Cll_p,
            self.Cll_r,
            self.Cll_beta,
        )
        Cn_p, Cn_r, Cn_beta = (
            self.Cn_p,
            self.Cn_r,
            self.Cn_beta,
        )
        Cy_del_R = self.Cy_del_R
        Cll_del_R, Cll_del_A = self.Cll_del_R, self.Cll_del_A
        Cm_del_E, Cm_del_A = self.Cm_del_E, self.Cm_del_A
        Cn_del_R, Cn_del_A = self.Cn_del_R, self.Cn_del_A
        S, S2, St, b, c, d = self.S, self.S2, self.St, self.b, self.c, self.d
        dela, dele, delr = dels
        CL, CD, CM = self.aero_coeff(alp)
        Fx = -qbar * S * CD
        Fy = qbar * S * (p * Cy_p + r * Cy_r + beta * Cy_beta + delr * Cy_del_R)
        Fz = -qbar * S * CL
        l = (
            qbar
            * b
            * (
                S * (p * Cll_p + r * Cll_r + beta * Cll_beta + delr * Cll_del_R)
                + S2 * dela * Cll_del_A
            )
        )
        m = qbar * c * S * (CM + dele * Cm_del_E + dela * Cm_del_A)
        n = (
            qbar
            * d
            * St
            * (p * Cn_p + r * Cn_r + beta * Cn_beta + delr * Cn_del_R + dela * Cn_del_A)
        )
        return np.vstack((Fx, Fy, Fz, l, m, n))

    def B_Gravity(self, quat):
        l = m = n = 0
        return np.vstack((quat2dcm(quat) @ (self.m * self.g * self.e3), l, m, n))

    def get_rho(self, altitude):
        pressure = 101325 * (1 - 2.25569e-5 * altitude) ** 5.25616
        temperature = 288.14 - 0.00649 * altitude
        return pressure / (287 * temperature)

    def aero_coeff(self, alp):
        CL = np.interp(alp, self.tables["alp"], self.tables["CL"])
        CD = np.interp(alp, self.tables["alp"], self.tables["CD"])
        Cm = np.interp(alp, self.tables["alp"], self.tables["Cm"])
        return np.vstack((CL, CD, Cm))

    def pwm2cmd(self, pwm):
        """
        pwm: 1000 ~ 2000
        cmd: 0 ~ 1
        """
        return (pwm - 1000) / 1000

    def cmd2pwm(self, cmd):
        """
        cmd: 0 ~ 1
        pwm: 1000 ~ 2000
        """
        return cmd * 1000 + 1000

    def get_trim_fixed(
        self,
        z0={
            "alpha": 0.0,
            "beta": 0,
            "pusher1": 1500,
            "pusher2": 1500,
            "dela": 0,
            "dele": 0,
            "delr": 0,
        },
        fixed={"h": 10, "VT": 10},
        method="SLSQP",
        options={"disp": False, "ftol": 1e-10},
        verbose=False,
    ):
        z0 = list(z0.values())
        fixed = list(fixed.values())
        bounds = (
            np.deg2rad((0, 20)),
            np.deg2rad((-10, 10)),
            self.control_limits["pwm"],
            self.control_limits["pwm"],
            self.control_limits["dela"],
            self.control_limits["dele"],
            self.control_limits["delr"],
        )
        result = scipy.optimize.minimize(
            self._trim_cost_fixed,
            z0,
            args=(fixed,),
            bounds=bounds,
            method=method,
            options=options,
        )

        h, VT = fixed
        if np.isclose(VT, 0):
            alp, beta, pusher1, pusher2, dela, dele, delr = 0, 0, 1000, 1000, 0, 0, 0
        else:
            alp, beta, pusher1, pusher2, dela, dele, delr = result.x
        pos_trim = np.vstack((0, 0, -h))
        vel_trim = np.vstack(
            (VT * cos(alp) * cos(beta), VT * sin(beta), VT * sin(alp) * cos(beta))
        )
        quat_trim = np.vstack(angle2quat(0, alp, 0))
        omega_trim = np.vstack((0, 0, 0))
        pwms_pusher = np.vstack((pusher1, pusher2))
        dels = np.vstack((dela, dele, delr))

        x_trims = (pos_trim, vel_trim, quat_trim, omega_trim)
        u_trims_fixed = (pwms_pusher, dels)
        return x_trims, u_trims_fixed

    def _trim_cost_fixed(self, z, fixed):
        h, VT = fixed
        alp, beta, pusher1, pusher2, dela, dele, delr = z
        pos_trim = np.vstack((0, 0, -h))
        vel_trim = np.vstack(
            (VT * cos(alp) * cos(beta), VT * sin(beta), VT * sin(alp) * cos(beta))
        )
        quat_trim = np.vstack(angle2quat(0, alp, 0))
        omega_trim = np.vstack((0, 0, 0))
        pwms_pusher = np.vstack((pusher1, pusher2))
        dels = np.vstack((dela, dele, delr))

        FM_Pusher = self.B_Pusher(pwms_pusher)
        FM_Fuselage = self.B_Fuselage(dels, pos_trim, vel_trim, omega_trim)
        FM_Gravity = self.B_Gravity(quat_trim)
        FM_Fixed = FM_Fuselage + FM_Pusher + FM_Gravity

        dots = self.deriv(pos_trim, vel_trim, quat_trim, omega_trim, FM_Fixed)
        dxs = np.append(dots[1], dots[3])
        weight = np.diag([10, 1, 1, 1000, 1000, 1000])
        return dxs.dot(weight).dot(dxs)

    def get_trim_vtol(
        self,
        z0={
            "rotor1": 1100,
            "rotor2": 1100,
            "rotor3": 1100,
            "rotor4": 1100,
            "rotor5": 1100,
            "rotor6": 1100,
        },
        fixed={
            "x_trim": (
                np.zeros((3, 1)),
                np.zeros((3, 1)),
                np.vstack((1, 0, 0, 0)),
                np.zeros((3, 1)),
            ),
            "u_trims_fixed": (1000 * np.ones((2, 1)), np.zeros((3, 1))),
        },
        method="SLSQP",
        options={"disp": False, "ftol": 1e-10},
        verbose=False,
    ):
        z0 = list(z0.values())
        fixed = list(fixed.values())
        bounds = (
            self.control_limits["pwm"],
            self.control_limits["pwm"],
            self.control_limits["pwm"],
            self.control_limits["pwm"],
            self.control_limits["pwm"],
            self.control_limits["pwm"],
        )
        result = scipy.optimize.minimize(
            self._trim_cost_vtol,
            z0,
            args=(fixed,),
            bounds=bounds,
            method=method,
            options=options,
        )

        rotor1, rotor2, rotor3, rotor4, rotor5, rotor6 = result.x
        pwms_rotor = np.vstack((rotor1, rotor2, rotor3, rotor4, rotor5, rotor6))

        u_trims_vtol = pwms_rotor
        return u_trims_vtol

    def _trim_cost_vtol(self, z, fixed):
        x_trims, u_trims_fixed = fixed
        rotor1, rotor2, rotor3, rotor4, rotor5, rotor6 = z
        pos_trim, vel_trim, quat_trim, omega_trim = x_trims
        pwms_pusher, dels = u_trims_fixed
        pwms_rotor = np.vstack((rotor1, rotor2, rotor3, rotor4, rotor5, rotor6))

        FM_VTOL = self.B_VTOL(pwms_rotor, omega_trim)
        FM_Pusher = self.B_Pusher(pwms_pusher)
        FM_Fuselage = self.B_Fuselage(dels, pos_trim, vel_trim, omega_trim)
        FM_Gravity = self.B_Gravity(quat_trim)
        FM = FM_VTOL + FM_Fuselage + FM_Pusher + FM_Gravity

        dots = self.deriv(pos_trim, vel_trim, quat_trim, omega_trim, FM)
        dxs = np.append(dots[1], dots[3])
        weight = np.diag([1, 1, 1, 1000, 1000, 1000])
        return dxs.dot(weight).dot(dxs)

    def saturate(self, ctrls):
        _ctrls = np.zeros((ctrls.shape))
        pwm_min, pwm_max = self.control_limits["pwm"]
        dela_min, dela_max = self.control_limits["dela"]
        dele_min, dele_max = self.control_limits["dele"]
        delr_min, delr_max = self.control_limits["delr"]
        _ctrls[:8] = np.clip(ctrls[:8], pwm_min, pwm_max)
        _ctrls[8] = np.clip(ctrls[8], dela_min, dela_max)
        _ctrls[9] = np.clip(ctrls[9], dele_min, dele_max)
        _ctrls[10] = np.clip(ctrls[10], delr_min, delr_max)
        return _ctrls


if __name__ == "__main__":
    system = LC62()
    pos, vel, quat, omega = system.x_trims
    pwms_pusher, dels = system.u_trims_fixed
    pwms_rotor = system.u_trims_vtol
    ctrls = np.vstack((pwms_rotor, pwms_pusher, dels))
    FM = system.get_FM(pos, vel, quat, omega, ctrls)
    system.set_dot(t=0, FM=FM)
    print(repr(system))
