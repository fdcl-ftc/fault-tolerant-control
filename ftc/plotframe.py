import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle, Polygon
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import animation

import fym
from fym.utils.rot import quat2dcm


class QUADFrame:
    def __init__(self, ax, xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2)):
        self.ax = ax
        self.xlim, self.ylim, self.zlim = xlim, ylim, zlim

        self.d = 0.315
        self.rr = 0.1  # radius of rotor
        self.rc = 0.01

        self.b1 = np.vstack([1.0, 0.0, 0.0])
        self.b2 = np.vstack([0.0, 1.0, 0.0])
        self.b3 = np.vstack([0.0, 0.0, 1.0])

        self.ax.set(xlim3d=self.xlim, xlabel="X")
        self.ax.set(ylim3d=self.ylim, ylabel="Y")
        self.ax.set(zlim3d=self.zlim, zlabel="Z")

        self.alp_list = [0.1, 0.5, 1]

    def draw_at(self, x=np.zeros((3, 1)), q=np.vstack([1, 0, 0, 0]), lamb=np.ones((4, 1))):
        self.ax.clear()
        R = quat2dcm(q)

        x1 = (x + R @ self.b1 * self.d).ravel()
        x2 = (x + R @ self.b2 * self.d).ravel()
        x3 = (x + R @ (- self.b1) * self.d).ravel()
        x4 = (x + R @ (- self.b2) * self.d).ravel()

        # Center of the quadrotor
        body = self.ax.add_patch(Circle((x[0], x[1]), self.rc, color='y'))
        art3d.pathpatch_2d_to_3d(body, z=x[2])

        # Fault
        alps = np.ones(4,)
        for i in range(len(alps)):
            if lamb[i] == 0:
                alps[i] = self.alp_list[0]
            elif lamb[i] == 1:
                alps[i] = self.alp_list[2]
            else:
                alps[i] = self.alp_list[1]

        # Rotor
        r1 = self.ax.add_patch(Circle((x1[0], x1[1]), self.rr, color='r', alpha=alps[0]))
        r2 = self.ax.add_patch(Circle((x2[0], x2[1]), self.rr, color='b', alpha=alps[1]))
        r3 = self.ax.add_patch(Circle((x3[0], x3[1]), self.rr, color='r', alpha=alps[2]))
        r4 = self.ax.add_patch(Circle((x4[0], x4[1]), self.rr, color='b', alpha=alps[3]))
        art3d.pathpatch_2d_to_3d(r1, z=x1[2])
        art3d.pathpatch_2d_to_3d(r2, z=x2[2])
        art3d.pathpatch_2d_to_3d(r3, z=x3[2])
        art3d.pathpatch_2d_to_3d(r4, z=x4[2])

        # Quadrotor arms
        _x = x.ravel()
        self.ax.plot([_x[0], x1[0]], [_x[1], x1[1]], [_x[2], x1[2]], "k")
        self.ax.plot([_x[0], x2[0]], [_x[1], x2[1]], [_x[2], x2[2]], "k")
        self.ax.plot([_x[0], x3[0]], [_x[1], x3[1]], [_x[2], x3[2]], "k")
        self.ax.plot([_x[0], x4[0]], [_x[1], x4[1]], [_x[2], x4[2]], "k")


class LC62Frame:
    def __init__(self, ax, xlim=(-5, 5), ylim=(-5, 5), zlim=(-5, 5)):
        self.ax = ax
        self.xlim, self.ylim, self.zlim = xlim, ylim, zlim

        self.dx1 = 0.9815
        self.dx2 = 0.0235
        self.dx3 = 1.1235
        self.dy = 0.717
        self.b1 = 1.45  # front wing span [m]
        self.b2 = 2.2  # rear wing span [m]
        self.c1 = 0.2624 / self.b1  # front wing chord [m]
        self.c2 = 0.5898 / self.b2  # rear wing chord [m]
        self.rr = 0.762 / 2  # radius of rotor
        self.rp = 0.525 / 2  # radius of pusher
        self.wf = 2.025  # width of fuselage
        self.hf = 0.350  # height of fuselage

        self.ax.set(xlim3d=self.xlim, xlabel="X")
        self.ax.set(ylim3d=self.ylim, ylabel="Y")
        self.ax.set(zlim3d=self.zlim, zlabel="Z")

        self.alp_list = [0.1, 0.5, 1]

    def draw_at(self, x=np.zeros((3, 1)), q=np.vstack([1, 0, 0, 0]), lamb=np.ones((11, 1))):
        self.ax.clear()
        R = quat2dcm(q)

        xr1 = (x + R @ np.vstack([self.dx1, self.dy, 0.0])).ravel()
        xr2 = (x + R @ np.vstack([self.dx2, self.dy, 0.0])).ravel()
        xr3 = (x + R @ np.vstack([-self.dx3, self.dy, 0.0])).ravel()
        xr4 = (x + R @ np.vstack([self.dx1, -self.dy, 0.0])).ravel()
        xr5 = (x + R @ np.vstack([self.dx2, -self.dy, 0.0])).ravel()
        xr6 = (x + R @ np.vstack([-self.dx3, -self.dy, 0.0])).ravel()

        xp1 = (x + R @ np.vstack([-self.dx3/2, self.b2/2, 0.0])).ravel()
        xp2 = (x + R @ np.vstack([-self.dx3/2, -self.b2/2, 0.0])).ravel()

        xfw1 = (x + R @ np.vstack([self.dx1/2+self.c1/2+0.01, 0, 0])).ravel()
        xfw2 = (x + R @ np.vstack([self.dx1/2+self.c1/2, self.b1/2, 0])).ravel()
        xfw3 = (x + R @ np.vstack([self.dx1/2-self.c1/2, self.b1/2, 0])).ravel()
        xfw4 = (x + R @ np.vstack([self.dx1/2-self.c1/2-0.1, 0, 0])).ravel()
        xfw5 = (x + R @ np.vstack([self.dx1/2-self.c1/2, -self.b1/2, 0])).ravel()
        xfw6 = (x + R @ np.vstack([self.dx1/2+self.c1/2, -self.b1/2, 0])).ravel()

        xrw1 = (x + R @ np.vstack([-self.dx3/2+self.c2/2+0.01, 0, 0])).ravel()
        xrw2 = (x + R @ np.vstack([-self.dx3/2+self.c2/2, self.b2/2, 0])).ravel()
        xrw3 = (x + R @ np.vstack([-self.dx3/2-self.c2/2, self.b2/2, 0])).ravel()
        xrw4 = (x + R @ np.vstack([-self.dx3/2-self.c2/2-0.1, 0, 0])).ravel()
        xrw5 = (x + R @ np.vstack([-self.dx3/2-self.c2/2, -self.b2/2, 0])).ravel()
        xrw6 = (x + R @ np.vstack([-self.dx3/2+self.c2/2, -self.b2/2, 0])).ravel()

        # Fuselage
        fuselage = self.ax.add_patch(Ellipse((x[0], x[1]), self.wf, self.hf, fc="0.5"))
        art3d.pathpatch_2d_to_3d(fuselage, z=x[2])

        # Wing
        fwxs = [xfw1[0], xfw2[0], xfw3[0], xfw4[0], xfw5[0], xfw6[0]]
        fwys = [xfw1[1], xfw2[1], xfw3[1], xfw4[1], xfw5[1], xfw6[1]]
        rwxs = [xrw1[0], xrw2[0], xrw3[0], xrw4[0], xrw5[0], xrw6[0]]
        rwys = [xrw1[1], xrw2[1], xrw3[1], xrw4[1], xrw5[1], xrw6[1]]
        fw = self.ax.add_patch(Polygon(list(zip(fwxs, fwys)), fc="0.5"))
        rw = self.ax.add_patch(Polygon(list(zip(rwxs, rwys)), fc="0.5"))
        art3d.pathpatch_2d_to_3d(fw, z=(xfw1[2]+xfw4[2])/2)
        art3d.pathpatch_2d_to_3d(rw, z=(xrw1[2]+xrw4[2])/2)

        # Fault
        alps = np.ones(11,)
        for i in range(len(alps)):
            if lamb[i] == 0:
                alps[i] = self.alp_list[0]
            elif lamb[i] == 1:
                alps[i] = self.alp_list[2]
            else:
                alps[i] = self.alp_list[1]

        # Rotor
        r1 = self.ax.add_patch(Circle((xr1[0], xr1[1]), self.rr, fc="tab:pink", ec="0.3", alpha=alps[2]))
        r2 = self.ax.add_patch(Circle((xr2[0], xr2[1]), self.rr, fc="tab:pink", ec="0.3", alpha=alps[1]))
        r3 = self.ax.add_patch(Circle((xr3[0], xr3[1]), self.rr, fc="tab:pink", ec="0.3", alpha=alps[5]))
        r4 = self.ax.add_patch(Circle((xr4[0], xr4[1]), self.rr, fc="tab:pink", ec="0.3", alpha=alps[4]))
        r5 = self.ax.add_patch(Circle((xr5[0], xr5[1]), self.rr, fc="tab:pink", ec="0.3", alpha=alps[0]))
        r6 = self.ax.add_patch(Circle((xr6[0], xr6[1]), self.rr, fc="tab:pink", ec="0.3", alpha=alps[3]))
        art3d.pathpatch_2d_to_3d(r1, z=xr1[2])
        art3d.pathpatch_2d_to_3d(r2, z=xr2[2])
        art3d.pathpatch_2d_to_3d(r3, z=xr3[2])
        art3d.pathpatch_2d_to_3d(r4, z=xr4[2])
        art3d.pathpatch_2d_to_3d(r5, z=xr5[2])
        art3d.pathpatch_2d_to_3d(r6, z=xr6[2])

        # Arms
        self.ax.plot([xr1[0], xr3[0]], [xr1[1], xr3[1]], [xr1[2], xr3[2]], "k")
        self.ax.plot([xr4[0], xr6[0]], [xr4[1], xr6[1]], [xr4[2], xr6[2]], "k")

        # Pusher
        p1 = self.ax.add_patch(Circle((xp1[1], xp1[2]), self.rp, fc="tab:purple", ec="0.3", alpha=alps[6]))
        p2 = self.ax.add_patch(Circle((xp2[1], xp2[2]), self.rp, fc="tab:purple", ec="0.3", alpha=alps[7]))
        art3d.pathpatch_2d_to_3d(p1, z=xp1[0], zdir="x")
        art3d.pathpatch_2d_to_3d(p2, z=xp2[0], zdir="x")


def update_plot(i, uav, t, x, q, lamb, numFrames=1):
    uav.draw_at(np.vstack(x[:, i*numFrames]), np.vstack(q[:, i*numFrames]), lamb[:, i*numFrames])

    uav.ax.set(xlim3d=uav.xlim, xlabel="X")
    uav.ax.set(ylim3d=uav.ylim, ylabel="Y")
    uav.ax.set(zlim3d=uav.zlim, zlabel="Z")

    titleTime = uav.ax.text2D(0.05, 0.95, "", transform=uav.ax.transAxes)
    titleTime.set_text(u"Time = {:.2f} s".format(t[i*numFrames]))


if __name__ == '__main__':
    data = fym.load("data.h5")["env"]
    t = data["t"]
    x = data["plant"]["pos"].squeeze(-1).T
    q = data["plant"]["quat"].squeeze(-1).T
    lamb = data["Lambda"].squeeze(-1).T

    numFrames = 10

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # uav = QUADFrame(ax)
    uav = LC62Frame(ax)
    ani = animation.FuncAnimation(
        fig, update_plot, frames=len(t[::numFrames]),
        fargs=(uav, t, x, q, lamb, numFrames), interval=1
    )
    # ani.save("animation.gif", dpi=80, writer="imagemagick", fps=25)

    plt.show()
