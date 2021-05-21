import numpy as np
import matplotlib.pyplot as plt

from multicopter import Multicopter

# testing list
test1 = 0 * np.ones((6, 1))
test2 = 50 * np.ones((6, 1))
r1, r2, r3, r4, r5, r6 = 0, 0, 0, 50, 0, 100
test3 = np.vstack([r1, r2, r3, r4, r5, r6])

pos = np.zeros((3, 1))
vel = np.zeros((3, 1))
quat = np.vstack([1, 0, 0, 0])
omeg = np.zeros((3, 1))

system = Multicopter(pos=pos, vel=vel, quat=quat, omega=omeg)

t = np.array([0])
dt = 0.01

for i in range(20):
    dpos, dvel, dquat, domeg = system.deriv(
        pos[:, -1:], vel[:, -1:], quat[:, -1:], omeg[:, -1:], test3)

    t = np.append(t, dt + t[-1])
    pos = np.hstack((pos, dt * dpos + pos[:, -1:]))
    vel = np.hstack((vel, dt * dvel + vel[:, -1:]))
    quat = np.hstack((quat, dt * dquat + quat[:, -1:]))
    omeg = np.hstack((omeg, dt * domeg + omeg[:, -1:]))

fig = plt.figure()
ax1 = fig.add_subplot(4, 1, 1)
ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)

ax1.plot(t, pos.T)
ax2.plot(t, vel.T)
ax3.plot(t, quat.T)
ax4.plot(t, np.rad2deg(omeg.T))

ax1.set_ylabel('Position')
ax1.legend([r'$x$', r'$y$', r'$z$'])

ax2.set_ylabel('Velocity')
ax2.legend([r'$u$', r'$v$', r'$w$'])

ax3.set_ylabel('Quaternion')
ax3.legend([r'$q_0$', r'$q_1$', r'$q_2$', r'$q_3$'])

ax4.set_ylabel('Angular Velocity')
ax4.legend([r'$p$', r'$q$', r'$r$'])
ax4.set_xlabel('Time [sec]')

plt.tight_layout()

plt.show()
