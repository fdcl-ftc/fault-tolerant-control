import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace as SN

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import angle2quat, quat2angle
import fym.logging

from ftc.models.multicopter import Multicopter
import ftc.agents.dsc as dsc
from ftc.agents.fdi import SimpleFDI
from ftc.faults.actuator import LoE, LiP, Float
from ftc.models.dryden import Wind


cfg = SN()


def load_config():
    cfg.env = SN()
    cfg.env.dt = 0.001
    cfg.env.max_t = 5

    cfg.act_dyn = SN()
    cfg.act_dyn.tau = 0.01

    cfg.FDI = SN()
    cfg.FDI.tau = 0.1

    cfg.CA = SN()
    cfg.CA.mu = 0.01
    cfg.CA.eps = 1e-8
    cfg.CA.W = np.eye(4)
    cfg.CA.alpha = 0.5
    cfg.CA.c = 0.9
    cfg.CA.eps = 1e-8
    cfg.CA.d_step = 0
    cfg.CA.gamma_step = 0

    dsc.load_config()
    dsc.cfg.K1 = np.diag([10, 10, 10, 10]) * 60
    dsc.cfg.K2 = np.diag([10, 10, 10, 10]) * 60
    dsc.cfg.Kbar = np.diag([10, 10, 10, 10]) * 60


class ActuatorDynamcs(BaseSystem):
    def __init__(self, tau, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau

    def set_dot(self, rotors, rotors_cmd):
        self.dot = - 1 / self.tau * (rotors - rotors_cmd)


class Env(BaseEnv):
    def __init__(self):
        super().__init__(**vars(cfg.env))
        self.plant = Multicopter(rtype="hexa-+")

        # Define actuator dynamics
        n = self.plant.mixer.B.shape[1]
        self.act_dyn = ActuatorDynamcs(cfg.act_dyn.tau, shape=(n, 1))

        # Set wind
        self.wind = Wind(cfg.env.dt, cfg.env.max_t)

        # Define FDI
        self.fdi = SimpleFDI(no_act=n, tau=cfg.FDI.tau)

        J, m, g, Jr, d, c = [
            getattr(self.plant, k) for k in ["J", "m", "g", "Jr", "d", "c"]]
        b = self.plant.mixer.b
        self.controller = dsc.DSCController(n, J, m, g, Jr, l=d, d=c, b=b)

        # Define faults
        self.sensor_faults = []
        self.actuator_faults = [
            LoE(time=3, index=1, level=0.5),
            LoE(time=5, index=2, level=0.2),
            LoE(time=7, index=1, level=0.1),
            Float(time=10, index=0),
        ]

    def step(self):
        *_, done = self.update()
        return done

    def get_cost(self, rotors, f, B, What):
        ut = f - B.dot(What).dot(rotors)

        return np.sum([
            ut.T.dot(cfg.CA.W).dot(ut),
            cfg.CA.mu * np.sum([(
                1 / (fi - cfg.CA.eps)**2
                + 1 / (self.plant.rotor_max - fi - cfg.CA.eps)**2
            ) for fi in rotors])
        ]) / 2

    def control_allocation(self, f, What):
        B = self.plant.mixer.B

        rotors = np.linalg.pinv(B.dot(What)).dot(f)
        rotors = np.clip(rotors, 0, self.plant.rotor_max)

        gamma = 1e-2

        for _ in range(cfg.CA.d_step):

            d = - What.T.dot(B.T).dot(cfg.CA.W.T).dot(
                f - B.dot(What).dot(rotors))

            if d.T.dot(d) < cfg.CA.eps:
                return rotors

            for _ in range(cfg.CA.gamma_step):
                rotors_next = rotors - gamma * d
                rotors_next = np.clip(rotors_next, 0, self.plant.rotor_max)

                J_next = self.get_cost(rotors_next, f, B, What)
                J = self.get_cost(rotors, f, B, What)

                if J_next < J - cfg.CA.alpha * gamma * d.T.dot(d):
                    break

                gamma = cfg.CA.c * gamma

        return rotors

    def get_ref(self, t):
        a1 = np.pi / 20
        a2 = np.pi / 2
        k1 = 2 * np.pi / 10
        k2 = 2 * np.pi / 20

        phid = a1 * np.sin(k1 * t)
        thetad = a1 * np.cos(k1 * t)
        psid = a2 * np.sin(k2 * t)
        zd = 1 + 0.5 * np.sin(k2 * t)

        phid_dot = a1 * k1 * np.cos(k1 * t)
        thetad_dot = -a1 * k1 * np.sin(k1 * t)
        psid_dot = a2 * k2 * np.cos(k2 * t)
        zd_dot = 0.5 * k2 * np.cos(k2 * t)

        xi1d = np.vstack((phid, thetad, psid, zd))
        xi1d_dot = np.vstack((phid_dot, thetad_dot, psid_dot, zd_dot))

        return xi1d, xi1d_dot

    def get_dl(self, mult_states, rotors, windvel):
        pos, vel, quat, omega = mult_states

        Omega = self.plant.get_Omega(rotors)
        F_wind, M_wind = self.plant.get_FM_wind(rotors, vel, omega, windvel)
        dl = np.vstack((
            np.diag(self.plant.kr).reshape(-1, 1),
            Omega,
            M_wind,
            F_wind[-1],
        ))
        return dl

    def _get_derivs(self, t, mult_states, xi2d, rotors, What):
        # Set sensor faults
        for sen_fault in self.sensor_faults:
            mult_states = sen_fault(t, mult_states)

        windvel = self.wind.get(t)
        dl_hat = self.get_dl(mult_states, rotors, windvel)
        xi1d, xi1d_dot = self.get_ref(t)
        forces, xi2d_dot, _ = self.controller.get_forces(
            t, mult_states, xi2d, windvel, dl_hat, xi1d, xi1d_dot)
        rotors_cmd = self.control_allocation(forces, What)

        # Set actuator faults
        for act_fault in self.actuator_faults:
            rotors = act_fault(t, rotors)

        W = self.fdi.get_true(rotors, rotors_cmd)

        return rotors_cmd, W, xi2d_dot, xi1d, xi1d_dot, forces

    def set_dot(self, t):
        mult_states = self.plant.observe_list()
        xi2d = self.controller.state
        What = self.fdi.state
        rotors = self.act_dyn.state

        rotors_cmd, W, xi2d_dot, *_ = self._get_derivs(
            t, mult_states, xi2d, rotors, What)

        self.plant.set_dot(t, rotors)
        self.fdi.set_dot(W)
        self.controller.set_dot(xi2d_dot)
        self.act_dyn.set_dot(rotors, rotors_cmd)

    def logger_callback(self, i, t, y, *args):
        states = self.observe_dict(y)
        mult_states = self.plant.observe_list(y)
        xi1 = np.vstack((
            np.vstack(quat2angle(states["plant"]["quat"])[::-1]),
            - states["plant"]["pos"][2]
        ))
        xi2d = states["controller"]["xi2d"]
        What = states["fdi"]
        rotors = states["act_dyn"]
        rotors_cmd, W, xi2d_dot, xi1d, xi1d_dot, forces = self._get_derivs(
            t, mult_states, xi2d, rotors, What)
        return dict(
            t=t, rotors_cmd=rotors_cmd, W=W,
            xi1=xi1, xi1d=xi1d, xi1d_dot=xi1d_dot, forces=forces, **states)


def run():
    env = Env()
    env.logger = fym.logging.Logger("data.h5")

    env.reset()

    while True:
        env.render()
        done = env.step()

        if done:
            break

    env.close()


def exp1():
    """Exp 1 runs DSC for a faulty multicopter."""
    load_config()
    run()


def exp1_plot():
    data = fym.logging.load("data.h5")

    plt.figure()

    ax = plt.subplot(411)
    plt.plot(data["t"], np.rad2deg(data["xi1"][:, :3, 0]), "k", label="Response")
    plt.plot(data["t"], np.rad2deg(data["xi1d"][:, :3, 0]), "r--", label="Command")
    plt.ylabel("Angles, deg")
    plt.legend()

    plt.subplot(412, sharex=ax)
    plt.plot(data["t"], data["xi1"][:, 3, 0], "k")
    plt.plot(data["t"], data["xi1d"][:, 3, 0], "r--")
    plt.ylabel("Altitude, m")

    plt.subplot(413, sharex=ax)
    plt.plot(data["t"], data["act_dyn"].squeeze(), "k")
    plt.plot(data["t"], data["rotors_cmd"].squeeze(), "r--")
    plt.ylabel("Rotors")

    plt.subplot(414, sharex=ax)
    plt.plot(data["t"], np.diagonal(data["fdi"], axis1=1, axis2=2),
             "r--", label="Est.")
    plt.plot(data["t"], np.diagonal(data["W"], axis1=1, axis2=2),
             "k", label="True")
    plt.ylabel("LoE")

    plt.xlabel("Time, sec")

    plt.show()

    plt.show()


if __name__ == "__main__":
    exp1()
    exp1_plot()
