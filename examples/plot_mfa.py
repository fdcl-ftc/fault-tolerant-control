import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform

import fym
from fym.utils.rot import angle2dcm


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
    ax.add_artist(arrow)


setattr(Axes3D, "arrow3D", _arrow3D)


def update_plot(
    i, ax, data, numFrames,
    Elim=(-2, 2), Nlim=(-2, 2), Ulim=(-2, 2),
    scale_F=0.01,
    scale_M=0.01,
):
    ax.clear()
    _i = i * numFrames
    t = data["t"]
    pos = data["posd"][_i, :, :]
    mfa = data["mfa"][_i]
    NED2ENU = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1],
    ])
    # ang = data["ang"][_i, :, :]  # Euler angle
    # dcm = angle2dcm(*ang)  # I (NED) to B (body)
    FM = data["FM"][_i, :, :]
    F = FM[0:3, :]
    M = FM[3:6, :]

    if mfa:
        colors = {
            "force": "#1f77b4",
            "torque": "#2ca02c",
        }
    else:
        colors = {
            "force": "#d62728",
            "torque": "#ff7f0e",
        }

    ax.scatter(*(NED2ENU @ pos).ravel(), color="black")
    ax.arrow3D(
        *(NED2ENU @ pos).ravel(),
        *(NED2ENU @ (pos + scale_F*F)).ravel(),
        edgecolor=colors["force"],
        facecolor=colors["force"],
        label=f"force (scale: {scale_F})",
    )
    ax.arrow3D(
        *(NED2ENU @ pos).ravel(),
        *(NED2ENU @ (pos + scale_M*M)).ravel(),
        edgecolor=colors["torque"],
        facecolor=colors["torque"],
        label=f"torque (scale: {scale_M})",
    )

    ax.set(xlim3d=Elim, xlabel="E")
    ax.set(ylim3d=Nlim, ylabel="N")
    ax.set(zlim3d=Ulim, zlabel="U")
    titleTime = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
    titleTime.set_text("Time = {:.2f} s".format(t[_i]))
    titleForce = ax.text2D(0.95, 0.95, "", transform=ax.transAxes, color=colors["force"])
    titleForce.set_text("Force scale: {:.2f}".format(scale_F))
    titleTorque = ax.text2D(0.95, 0.90, "", transform=ax.transAxes, color=colors["torque"])
    titleTorque.set_text("Torque scale: {:.2f}".format(scale_M))


def main(args, numFrames=10):
    # data extraction
    data = fym.load(args.data)["env"]
    print(f"{args.data} is loaded for MFA visualization.")
    # plot figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ani = animation.FuncAnimation(
        fig,
        update_plot,
        frames=len(data["t"][::numFrames]),
        fargs=(ax, data, numFrames),
        interval=100,
    )
    if args.save:
        ani.save("animation.gif", dpi=80, writer="imagemagick", fps=10)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="data.h5")
    parser.add_argument("-s", "--save", action="store_true")
    args = parser.parse_args()
    main(args)
