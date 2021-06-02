import numpy as np
import matplotlib.pyplot as plt

from fym.utils.rot import quat2angle, angle2quat
import fym.logging
from fym.core import BaseEnv, BaseSystem

from ftc.models.multicopter import Multicopter
from ftc.agents.integral_slidingmode import IntegralSMC


class Env(BaseEnv):
    def __init__(self,
                 pos=np.zeros((3, 1)),
                 vel=np.zeros((3, 1)),
                 quat=np.vstack((1, 0, 0, 0)),
                 omega=np.zeros((3, 1)),
                 rtype="hexa-x"):
        super().__init__(dt=0.001, max_t=10)
        self.plant = Multicopter(pos, vel, quat, omega, rtype)

    def reset(self):
        super().reset()
        return self.observe_flat()

    def get_forces(self, action):
        return action

    def step(self, action):
        self.action = action
        t = self.clock.get()
        states = self.observe_flat()
        info = self.observe_dict()

        forces = self.get_forces(action)
        rotors_cmd = self.plant.mixer(forces)
        rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)
        *_, done = self.update()
        return t, states, info, rotors, done

    def set_dot(self, t):
        forces = self.get_forces(self.action)
        rotors_cmd = self.plant.mixer(forces)
        rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)
        self.plant.set_dot(t, rotors)


def run(env, pos, quat, ref, K, Kc, PHI, agent=None):
    obs = env.reset()
    logger = fym.logging.Logger(path='data.h5')
    action = np.zeros((4, 1))
    t = 0

    while True:
        env.render()

        if agent is None:
            action = 0
        else:
            action, sliding_surface = agent.get_action(obs, action, ref, K, Kc, PHI, t)

        t, next_obs, info, rotors, done = env.step(action)
        obs = next_obs
        logger.record(t=t, **info, rotors=rotors, sliding_surface=sliding_surface)

        if done:
            break

    env.close()
    logger.close()


def test_integralSMC(pos, quat, ic, ref, ref0, K, Kc, PHI):
    env = Env(pos=pos, quat=quat)
    agent = IntegralSMC(env, ic, ref0)
    run(env, pos, quat, ref, K, Kc, PHI, agent)
    plot_var()
    plt.show()


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

    fig3 = plt.figure()
    ax1 = fig3.add_subplot(6, 1, 1)
    ax2 = fig3.add_subplot(6, 1, 2, sharex=ax1)
    ax3 = fig3.add_subplot(6, 1, 3, sharex=ax1)
    ax4 = fig3.add_subplot(6, 1, 4, sharex=ax1)

    ax1.plot(data['t'], data['sliding_surface'].squeeze()[:, 0])
    ax2.plot(data['t'], data['sliding_surface'].squeeze()[:, 1])
    ax3.plot(data['t'], data['sliding_surface'].squeeze()[:, 2])
    ax4.plot(data['t'], data['sliding_surface'].squeeze()[:, 3])

    ax1.set_ylabel('s1')
    ax1.grid(True)
    ax2.set_ylabel('s2')
    ax2.grid(True)
    ax3.set_ylabel('s3')
    ax3.grid(True)
    ax4.set_ylabel('s4')
    ax4.grid(True)

    plt.tight_layout()


if __name__ == "__main__":
    # reference
    rpos = np.vstack((0, 0, -20))
    rvel = np.zeros((3, 1))
    rquat = np.vstack((1, 0, 0, 0))
    romega = np.zeros((3, 1))
    ref = np.vstack((rpos, rvel, rquat, romega))
    ref0 = np.vstack((rpos, rvel, rquat, romega))
    # perturbation
    pos_pertb = np.vstack([0, 0, -10])
    yaw = 0
    pitch = 5
    roll = 5
    quat_pertb = angle2quat(*np.deg2rad([yaw, pitch, roll]))
    ic = np.vstack((pos_pertb, np.zeros((3, 1)), quat_pertb, np.zeros((3, 1))))
    # gain
    K = np.array([[25, 20],
                  [200, 10],
                  [200, 10],
                  [25, 10]])
    Kc = np.vstack((1, 1, 1, 1))
    PHI = np.vstack([1] * 4)
    test_integralSMC(pos_pertb, quat_pertb, ic, ref, ref0, K, Kc, PHI)
