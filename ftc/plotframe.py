import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
from fym.utils.rot import angle2quat, quat2dcm
from matplotlib import animation
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform


def _transform_zdir(zdir):
    zdir = art3d.get_dir_vector(zdir)
    zn = zdir / np.linalg.norm(zdir)

    cos_angle = zn[2]
    sin_angle = np.linalg.norm(zn[:2])
    if sin_angle == 0:
        return np.sign(cos_angle) * np.eye(3)

    d = np.array((zn[1], -zn[0], 0))
    d /= sin_angle
    ddt = np.outer(d, d)
    skew = np.array([[0, 0, -d[1]], [0, 0, d[0]], [d[1], -d[0], 0]], dtype=np.float64)
    return ddt + cos_angle * (np.eye(3) - ddt) + sin_angle * skew


def set_3d_properties(self, verts, zs=0, zdir="z"):
    zs = np.broadcast_to(zs, len(verts))
    self._segment3d = np.asarray(
        [
            np.dot(_transform_zdir(zdir), (x, y, 0)) + (0, 0, z)
            for ((x, y), z) in zip(verts, zs)
        ]
    )


def pathpatch_translate(pathpatch, delta):
    pathpatch._segment3d += np.asarray(delta)


art3d.Patch3D.set_3d_properties = set_3d_properties
art3d.Patch3D.translate = pathpatch_translate


def to_3d(pathpatch, z=0.0, zdir="z", delta=(0, 0, 0)):
    if not hasattr(pathpatch.axes, "get_zlim"):
        raise ValueError("Axes projection must be 3D")
    art3d.pathpatch_2d_to_3d(pathpatch, z=z, zdir=zdir)
    pathpatch.translate(delta)
    return pathpatch


matplotlib.patches.Patch.to_3d = to_3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add an 3d arrow to an `Axes3D` instance."""

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    arrow.set_mutation_scale(15)
    arrow.set_edgecolor("red")
    arrow.set_facecolor("red")
    ax.add_artist(arrow)


setattr(Axes3D, "arrow3D", _arrow3D)


class QUADFrame:
    def __init__(self, ax, xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2)):
        self.ax = ax
        self.xlim, self.ylim, self.zlim = xlim, ylim, zlim

        self.d = 0.315
        self.rr = 0.1  # radius of rotor
        self.rc = 0.01

        self.b1 = np.array([1.0, 0.0, 0.0])
        self.b2 = np.array([0.0, 1.0, 0.0])
        self.b3 = np.array([0.0, 0.0, 1.0])

        self.ax.set(xlim3d=self.xlim, xlabel="E")
        self.ax.set(ylim3d=self.ylim, ylabel="N")
        self.ax.set(zlim3d=self.zlim, zlabel="U")

    def draw_at(
        self,
        x=np.zeros((3, 1)),
        u=np.ones((4, 1)),
        q=np.vstack((1, 0, 0, 0)),
        lamb=np.ones((4, 1)),
        wu=np.eye(4),
    ):
        self.ax.clear()
        R = quat2dcm(q)

        Rc = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])  # NED to ENU
        _x = Rc @ x.ravel()
        x1 = _x + Rc @ R @ self.b2 * self.d
        x2 = _x + Rc @ R @ -self.b2 * self.d
        x3 = _x + Rc @ R @ self.b1 * self.d
        x4 = _x + Rc @ R @ -self.b1 * self.d
        e3 = tuple(Rc @ R @ self.b3)

        # Center of the quadrotor
        self.ax.add_patch(Circle((0, 0), self.rc, color="y")).to_3d(zdir=e3, delta=_x)

        # Fault
        alps = lamb.ravel()

        # Rotor
        self.ax.add_patch(Circle((0, 0), self.rr, color="r", alpha=alps[0])).to_3d(
            zdir=e3, delta=x1
        )
        self.ax.add_patch(Circle((0, 0), self.rr, color="r", alpha=alps[1])).to_3d(
            zdir=e3, delta=x2
        )
        self.ax.add_patch(Circle((0, 0), self.rr, color="b", alpha=alps[2])).to_3d(
            zdir=e3, delta=x3
        )
        self.ax.add_patch(Circle((0, 0), self.rr, color="b", alpha=alps[3])).to_3d(
            zdir=e3, delta=x4
        )

        # Quadrotor arms
        self.ax.plot([_x[0], x1[0]], [_x[1], x1[1]], [_x[2], x1[2]], "k")
        self.ax.plot([_x[0], x2[0]], [_x[1], x2[1]], [_x[2], x2[2]], "k")
        self.ax.plot([_x[0], x3[0]], [_x[1], x3[1]], [_x[2], x3[2]], "k")
        self.ax.plot([_x[0], x4[0]], [_x[1], x4[1]], [_x[2], x4[2]], "k")

        # Arrow
        dar = (wu @ -u) * np.vstack((e3, e3, e3, e3))
        self.ax.arrow3D(x1[0], x1[1], x1[2], dar[0, 0], dar[0, 1], dar[0, 2])
        self.ax.arrow3D(x2[0], x2[1], x2[2], dar[1, 0], dar[1, 1], dar[1, 2])
        self.ax.arrow3D(x3[0], x3[1], x3[2], dar[2, 0], dar[2, 1], dar[2, 2])
        self.ax.arrow3D(x4[0], x4[1], x4[2], dar[3, 0], dar[3, 1], dar[3, 2])


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

        self.ax.set(xlim3d=self.xlim, xlabel="E")
        self.ax.set(ylim3d=self.ylim, ylabel="N")
        self.ax.set(zlim3d=self.zlim, zlabel="U")

    def draw_at(
        self,
        x=np.zeros((3, 1)),
        u=np.ones((11, 1)),
        q=np.vstack((1, 0, 0, 0)),
        lamb=np.ones((11, 1)),
        wu=np.eye(11),
    ):
        self.ax.clear()
        R = quat2dcm(q)
        Rc = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])  # NED to ENU
        _x = Rc @ x.ravel()
        e1 = tuple(Rc @ R @ np.array([1, 0, 0]))
        e3 = tuple(Rc @ R @ np.array([0, 0, 1]))

        xr1 = _x + Rc @ R @ np.array([self.dx1, self.dy, 0.0])
        xr2 = _x + Rc @ R @ np.array([self.dx2, self.dy, 0.0])
        xr3 = _x + Rc @ R @ np.array([-self.dx3, self.dy, 0.0])
        xr4 = _x + Rc @ R @ np.array([self.dx1, -self.dy, 0.0])
        xr5 = _x + Rc @ R @ np.array([self.dx2, -self.dy, 0.0])
        xr6 = _x + Rc @ R @ np.array([-self.dx3, -self.dy, 0.0])

        xp1 = _x + Rc @ R @ np.array([-self.dx3 / 2, self.b2 / 2, 0.0])
        xp2 = _x + Rc @ R @ np.array([-self.dx3 / 2, -self.b2 / 2, 0.0])

        # Fuselage
        dff = 0.1
        dfr = 0.5
        xf1 = _x + Rc @ R @ np.array([self.dx1 + dff, 0, 0])
        xf2 = _x + Rc @ R @ np.array([self.dx1, self.hf / 2, 0])
        xf3 = _x + Rc @ R @ np.array([-self.dx3 / 2, self.hf / 2, 0])
        xf4 = _x + Rc @ R @ np.array([-self.dx3 / 2 - dfr, 0, 0])
        xf5 = _x + Rc @ R @ np.array([-self.dx3 / 2, -self.hf / 2, 0])
        xf6 = _x + Rc @ R @ np.array([self.dx1, -self.hf / 2, 0])

        fxs = [xf1[0], xf2[0], xf3[0], xf4[0], xf5[0], xf6[0]]
        fys = [xf1[1], xf2[1], xf3[1], xf4[1], xf5[1], xf6[1]]
        fzs = [xf1[2], xf2[2], xf3[2], xf4[2], xf5[2], xf6[2]]
        self.ax.add_collection3d(Poly3DCollection([list(zip(fxs, fys, fzs))], fc="0.5"))

        # Forward wing
        dfw = 0.1
        _xfw = _x + Rc @ R @ np.array([self.dx1 / 2, 0, 0])
        xfw1 = _xfw + Rc @ R @ np.array([self.c1 / 2, self.b1 / 2 - dfw, 0])
        xfw2 = _xfw + Rc @ R @ np.array([0, self.b1 / 2, 0])
        xfw3 = _xfw + Rc @ R @ np.array([-self.c1 / 2, self.b1 / 2 - dfw, 0])
        xfw4 = _xfw + Rc @ R @ np.array([-self.c1 / 2, -self.b1 / 2 + dfw, 0])
        xfw5 = _xfw + Rc @ R @ np.array([0, -self.b1 / 2, 0])
        xfw6 = _xfw + Rc @ R @ np.array([self.c1 / 2, -self.b1 / 2 + dfw, 0])

        fwxs = [xfw1[0], xfw2[0], xfw3[0], xfw4[0], xfw5[0], xfw6[0]]
        fwys = [xfw1[1], xfw2[1], xfw3[1], xfw4[1], xfw5[1], xfw6[1]]
        fwzs = [xfw1[2], xfw2[2], xfw3[2], xfw4[2], xfw5[2], xfw6[2]]
        self.ax.add_collection3d(
            Poly3DCollection([list(zip(fwxs, fwys, fwzs))], fc="0.5")
        )

        # Rear wing
        drw = 0.1
        _xrw = _x + Rc @ R @ np.array([-self.dx3 / 2, 0, 0])
        xrw1 = _xrw + Rc @ R @ np.array([self.c2 / 2, self.b2 / 2 - drw, 0])
        xrw2 = _xrw + Rc @ R @ np.array([0, self.b2 / 2, 0])
        xrw3 = _xrw + Rc @ R @ np.array([-self.c2 / 2, self.b2 / 2 - drw, 0])
        xrw4 = _xrw + Rc @ R @ np.array([-self.c2 / 2, -self.b2 / 2 + drw, 0])
        xrw5 = _xrw + Rc @ R @ np.array([0, -self.b2 / 2, 0])
        xrw6 = _xrw + Rc @ R @ np.array([self.c2 / 2, -self.b2 / 2 + drw, 0])

        rwxs = [xrw1[0], xrw2[0], xrw3[0], xrw4[0], xrw5[0], xrw6[0]]
        rwys = [xrw1[1], xrw2[1], xrw3[1], xrw4[1], xrw5[1], xrw6[1]]
        rwzs = [xrw1[2], xrw2[2], xrw3[2], xrw4[2], xrw5[2], xrw6[2]]
        self.ax.add_collection3d(
            Poly3DCollection([list(zip(rwxs, rwys, rwzs))], fc="0.5")
        )

        # Fault
        alps = lamb.ravel()

        # Rotor
        self.ax.add_patch(
            Circle((0, 0), self.rr, fc="tab:pink", ec="0.3", alpha=alps[2])
        ).to_3d(zdir=e3, delta=xr1)
        self.ax.add_patch(
            Circle((0, 0), self.rr, fc="tab:pink", ec="0.3", alpha=alps[1])
        ).to_3d(zdir=e3, delta=xr2)
        self.ax.add_patch(
            Circle((0, 0), self.rr, fc="tab:pink", ec="0.3", alpha=alps[5])
        ).to_3d(zdir=e3, delta=xr3)
        self.ax.add_patch(
            Circle((0, 0), self.rr, fc="tab:pink", ec="0.3", alpha=alps[4])
        ).to_3d(zdir=e3, delta=xr4)
        self.ax.add_patch(
            Circle((0, 0), self.rr, fc="tab:pink", ec="0.3", alpha=alps[0])
        ).to_3d(zdir=e3, delta=xr5)
        self.ax.add_patch(
            Circle((0, 0), self.rr, fc="tab:pink", ec="0.3", alpha=alps[3])
        ).to_3d(zdir=e3, delta=xr6)

        # Arms
        self.ax.plot([xr1[0], xr3[0]], [xr1[1], xr3[1]], [xr1[2], xr3[2]], "k")
        self.ax.plot([xr4[0], xr6[0]], [xr4[1], xr6[1]], [xr4[2], xr6[2]], "k")

        # Pusher
        self.ax.add_patch(
            Circle((0, 0), self.rp, fc="tab:purple", ec="0.3", alpha=alps[6])
        ).to_3d(zdir=e1, delta=xp1)
        self.ax.add_patch(
            Circle((0, 0), self.rp, fc="tab:purple", ec="0.3", alpha=alps[7])
        ).to_3d(zdir=e1, delta=xp2)

        # Arrow for Rotor
        dar = (
            wu[:6, :6] @ -np.vstack((u[2], u[1], u[5], u[4], u[0], u[3]))
        ) * np.vstack((e3, e3, e3, e3, e3, e3))
        self.ax.arrow3D(xr1[0], xr1[1], xr1[2], dar[0, 0], dar[0, 1], dar[0, 2])
        self.ax.arrow3D(xr2[0], xr2[1], xr2[2], dar[1, 0], dar[1, 1], dar[1, 2])
        self.ax.arrow3D(xr3[0], xr3[1], xr3[2], dar[2, 0], dar[2, 1], dar[2, 2])
        self.ax.arrow3D(xr4[0], xr4[1], xr4[2], dar[3, 0], dar[3, 1], dar[3, 2])
        self.ax.arrow3D(xr5[0], xr5[1], xr5[2], dar[4, 0], dar[4, 1], dar[4, 2])
        self.ax.arrow3D(xr6[0], xr6[1], xr6[2], dar[5, 0], dar[5, 1], dar[5, 2])


def update_plot(i, uav, t, x, u, q, lamb, wu, numFrames=1):
    uav.draw_at(
        np.vstack(x[i * numFrames, :]),
        np.vstack(u[i * numFrames, :]),
        np.vstack(q[i * numFrames, :]),
        lamb[i * numFrames, :],
        wu,
    )

    posx, posy, posz = x[i * numFrames, :]
    uav.ax.set(xlim3d=(posx + uav.xlim[0], posx + uav.xlim[1]), xlabel="E")
    uav.ax.set(ylim3d=(posy + uav.ylim[0], posy + uav.ylim[1]), ylabel="N")
    uav.ax.set(zlim3d=(posz + uav.zlim[0], posz + uav.zlim[1]), zlabel="U")

    titleTime = uav.ax.text2D(0.05, 0.95, "", transform=uav.ax.transAxes)
    titleTime.set_text("Time = {:.2f} s".format(t[i * numFrames]))


if __name__ == "__main__":
    vehicle = ""

    t = np.arange(0, 20, 0.01)
    x = q = []
    for i in t:
        if i == 0:
            x = np.vstack((i / 20, i / 20, -1)).T
            q = np.vstack((1, 0, 0, 0)).T
        else:
            x = np.append(x, np.vstack((i / 2, i / 2, -1)).T, axis=0)
            q = np.append(
                q,
                angle2quat(*np.deg2rad(np.array([100 * i, 2 * i, 2 * i]))).T,
                axis=0,
            )

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    if vehicle == "quad":
        u = np.ones((len(t), 4))
        u[:, 3] = 0
        lamb = np.ones((len(t), 4))
        lamb[:, 3] = 0
        wu = np.eye(4)

        uav = QUADFrame(ax)
    else:
        u = np.ones((len(t), 11))
        u[:, 3] = 0
        lamb = np.ones((len(t), 11))
        lamb[:, 3] = 0
        wu = np.eye(11)

        uav = LC62Frame(ax, xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))

    numFrames = 10

    ani = animation.FuncAnimation(
        fig,
        update_plot,
        frames=len(t[::numFrames]),
        fargs=(uav, t, x, u, q, lamb, wu, numFrames),
        interval=1,
    )
    # ani.save("animation.gif", dpi=80, writer="imagemagick", fps=25)

    plt.show()
