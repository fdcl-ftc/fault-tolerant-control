import numpy as np
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
from fym.utils.linearization import jacob_analytic
from fym.utils.rot import angle2quat, quat2angle
import fym.logging
from fym.agents import LQR

from ftc.models.multicopter import Multicopter


class Env(BaseEnv):
    def __init__(self,
                 pos=np.zeros((3, 1)),
                 vel=np.zeros((3, 1)),
                 quat=np.vstack((1, 0, 0, 0)),
                 omega=np.zeros((3, 1)),
                 ref=np.vstack([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
                 rtype="hexa-x"):
        super().__init__(dt=0.001, max_t=10)
        self.plant = Multicopter(pos, vel, quat, omega, rtype)

        angle_ref = np.vstack(quat2angle(ref[6:10])[::-1])
        self.ref = np.vstack((ref[0:6], angle_ref, ref[10::]))

        m, g = self.plant.m, self.plant.g
        n_rotors = self.plant.mixer.B.shape[1]
        trim_rotors = np.vstack([m * g / n_rotors] * n_rotors)
        self.trim_forces = self.plant.mixer.inverse(trim_rotors)

        self.A = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.B = np.array([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [-1/m, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, self.plant.Jinv[0, 0], 0, 0],
                           [0, 0, self.plant.Jinv[1, 1], 0],
                           [0, 0, 0, self.plant.Jinv[2, 2]]])
        # C = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
        Q = np.diag([5, 5, 5, 0, 0, 0, 5, 5, 5, 0, 0, 0])
        R = np.diag([1, 1, 1, 1])
        self.K, *_ = LQR.clqr(self.A, self.B, Q, R)

    def observation(self, pos, vel, quat, omega):
        return np.vstack((pos, vel, np.vstack(quat2angle(quat)[::-1]), omega))

    def get_forces(self, pos, vel, quat, omega):
        x = self.observation(pos, vel, quat, omega)
        forces = self.trim_forces - self.K.dot(x - self.ref)
        return forces

    def step(self):
        t = self.clock.get()
        states_list = self.plant.observe_list()
        states = self.observe_dict()
        forces = self.get_forces(*states_list)
        rotors = self.plant.mixer(forces)
        *_, done = self.update()
        return t, states, rotors, done

    def set_dot(self, t):
        states_list = self.plant.observe_list()
        forces = self.get_forces(*states_list)
        rotors = self.plant.mixer(forces)
        self.plant.set_dot(t, rotors)


def run(pos, quat, ref):
    env = Env(pos=pos, quat=quat, ref=ref)
    env.reset()
    logger = fym.logging.Logger(path='data.h5')

    while True:
        env.render()
        t, states, rotors, done = env.step()
        logger.record(t=t, **states, rotors=rotors)

        if done:
            break

    env.close()
    logger.close()


def plot_var():
    data = fym.logging.load('data.h5')
    fig = plt.figure()

    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
    ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)

    ax1.plot(data['t'], data['plant']['pos'].squeeze())
    ax2.plot(data['t'], data['plant']['vel'].squeeze())
    ax3.plot(data['t'], data['plant']['quat'].squeeze())
    ax4.plot(data['t'], data['plant']['omega'].squeeze())

    ax1.set_ylabel('Position')
    ax1.legend([r'$x$', r'$y$', r'$z$'])
    ax1.grid(True)

    ax2.set_ylabel('Velocity')
    ax2.legend([r'$u$', r'$v$', r'$w$'])
    ax2.grid(True)

    ax3.set_ylabel('Quaternion')
    ax3.legend([r'$q_0$', r'$q_1$', r'$q_2$', r'$q_3$'])
    ax3.grid(True)

    ax4.set_ylabel('Angular Velocity')
    ax4.legend([r'$p$', r'$q$', r'$r$'])
    ax4.set_xlabel('Time [sec]')
    ax4.grid(True)

    plt.tight_layout()

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(6, 1, 1)
    ax2 = fig2.add_subplot(6, 1, 2, sharex=ax1)
    ax3 = fig2.add_subplot(6, 1, 3, sharex=ax1)
    ax4 = fig2.add_subplot(6, 1, 4, sharex=ax1)
    ax5 = fig2.add_subplot(6, 1, 5, sharex=ax1)
    ax6 = fig2.add_subplot(6, 1, 6, sharex=ax1)

    ax1.plot(data['t'], data['rotors'].squeeze()[:, 0])
    ax2.plot(data['t'], data['rotors'].squeeze()[:, 1])
    ax3.plot(data['t'], data['rotors'].squeeze()[:, 2])
    ax4.plot(data['t'], data['rotors'].squeeze()[:, 3])
    ax5.plot(data['t'], data['rotors'].squeeze()[:, 4])
    ax6.plot(data['t'], data['rotors'].squeeze()[:, 5])

    ax1.set_ylabel('rotor1')
    ax1.grid(True)
    ax2.set_ylabel('rotor2')
    ax2.grid(True)
    ax3.set_ylabel('rotor3')
    ax3.grid(True)
    ax4.set_ylabel('rotor4')
    ax4.grid(True)
    ax5.set_ylabel('rotor5')
    ax5.grid(True)
    ax6.set_ylabel('rotor6')
    ax6.grid(True)
    ax6.set_xlabel('Time [sec]')

    plt.tight_layout()


# input값 조절 가능
if __name__ == "__main__":
    # ref
    x = 0
    y = 0
    z = 50
    ref = np.vstack([x, y, -z, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    # perturbation
    pos_pertb = np.vstack([0, 0, 0])
    yaw = 10
    pitch = 0
    roll = 0
    quat_pertb = angle2quat(*np.deg2rad([yaw, pitch, roll]))
    run(pos=pos_pertb, quat=quat_pertb, ref=ref)
    plot_var()
    plt.show()
