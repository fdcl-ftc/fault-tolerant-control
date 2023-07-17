import matplotlib.colors as mpl_colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def color_fader(c1, c2, N):
    c1 = np.array(mpl_colors.to_rgb(c1))
    c2 = np.array(mpl_colors.to_rgb(c2))
    return [
        mpl_colors.to_hex((1 - i / (N - 1)) * c1 + i / (N - 1) * c2) for i in range(N)
    ]


class PolytopeDeterminer:
    """
        PolytopeDeterminer

    A mission-success determiner using polytope.
    """

    def __init__(
        self,
        u_min,
        u_max,
        allocator,
        scaling_factor=1.0,
    ):
        """
        u_min: (m,) array; minimum input (element-wise)
        u_max: (m,) array; maximum input (element-wise)
        allocator: A control allocation law supposed to be used in a controller of interest

        Here, input space `U` is defined as `u_min <= u <= u_max` (element-wise).
        """
        self.u_min = u_min
        self.u_max = u_max
        self.allocator = allocator
        self.scaling_factor = scaling_factor

    def get_lower_bound(self, lmbd):
        return self.scaling_factor * np.diag(lmbd) @ self.u_min

    def get_upper_bound(self, lmbd):
        return self.scaling_factor * np.diag(lmbd) @ self.u_max

    def determine_is_in(self, nu, lmbd):
        """
            determine_is_in

        Determine if the control input (will be allocated) is in input space U for given generalized force `nu` and actuator fault information `lmbd`.

        nu = [F, M^T]^T: (4,) array; generalized force
        B: (4 x m) array; control effectiveness matrix
        lmbd: (m,) array containing actuator fault information, ranging from 0 to 1 (effectiveness).
        """
        u = self.allocator(nu)
        is_larger_than_min = u >= self.get_lower_bound(lmbd)
        is_smaller_than_max = u <= self.get_upper_bound(lmbd)
        is_in = np.all(is_larger_than_min & is_smaller_than_max)
        return is_in

    def determine_are_in(self, nus, lmbds):
        """
        Apply `determine_is_in` for multiple pairs of (nu, lmbd)'s.
        """
        are_in = [self.determine_is_in(nu, lmbd) for (nu, lmbd) in zip(nus, lmbds)]
        return are_in

    def determine(self, *args, **kwargs):
        """
        Is the mission success? (by considering all points along a trajectory imply that corresponding contrl inputs are in the input space.)
        """
        are_in = self.determine_are_in(*args, **kwargs)
        return np.all(are_in)

    def create_palette(self):
        fig, axs = plt.subplots(3, 2)
        ax = axs[0, 0]
        ax.set_xlabel("u_1")
        ax.set_ylabel("u_2")
        ax = axs[0, 1]
        ax.set_xlabel("u_1")
        ax.set_ylabel("u_3")
        ax = axs[1, 0]
        ax.set_xlabel("u_1")
        ax.set_ylabel("u_4")
        ax = axs[1, 1]
        ax.set_xlabel("u_2")
        ax.set_ylabel("u_3")
        ax = axs[2, 0]
        ax.set_xlabel("u_2")
        ax.set_ylabel("u_4")
        ax = axs[2, 1]
        ax.set_xlabel("u_3")
        ax.set_ylabel("u_4")
        return fig, axs

    def visualize(self, *args, **kwargs):
        fig, axs = self.create_palette()
        return self._visualize(fig, axs, *args, **kwargs)

    def _visualize(
        self,
        fig,
        axs,
        nus,
        lmbds,
        color1="#00ff00",
        color2="#3575D5",
        color3="#ffa500",
        color4="#FF0000",
    ):
        colors_good = color_fader(color1, color2, len(lmbds))
        colors_bad = color_fader(color3, color4, len(lmbds))

        colors = [
            color_good if self.determine_is_in(nu, lmbd) else color_bad
            for (color_good, color_bad, nu, lmbd) in zip(
                colors_good, colors_bad, nus, lmbds
            )
        ]
        fig, axs = self._draw_bounds(fig, axs, lmbds, colors)
        fig, axs = self._draw_inputs(fig, axs, nus, lmbds, colors)
        return fig, axs

    def _draw_input(self, fig, axs, nu, color, marker):
        u = self.allocator(nu)
        u1, u2, u3, u4 = u
        axs[0, 0].scatter(
            u1,
            u2,
            color=color,
            marker=marker,
        )
        axs[0, 1].scatter(
            u1,
            u3,
            color=color,
            marker=marker,
        )
        axs[1, 0].scatter(
            u1,
            u4,
            color=color,
            marker=marker,
        )
        axs[1, 1].scatter(
            u2,
            u3,
            color=color,
            marker=marker,
        )
        axs[2, 0].scatter(
            u2,
            u4,
            color=color,
            marker=marker,
        )
        axs[2, 1].scatter(
            u3,
            u4,
            color=color,
            marker=marker,
        )
        return fig, axs

    def _draw_inputs(self, fig, axs, nus, lmbds, colors):
        markers = ["_" for _ in range(len(nus))]
        markers[0] = "o"
        markers[-1] = "o"
        for nu, lmbd, color, marker in zip(nus, lmbds, colors, markers):
            fig, axs = self._draw_input(fig, axs, nu, color, marker)
        return fig, axs

    def _draw_bounds(self, fig, axs, lmbds, colors, alpha=0.5):
        linewidths = [0.5 for _ in range(len(lmbds))]
        linewidths[0] = 2.0
        linewidths[-1] = 2.0
        for lmbd, color, lw in zip(lmbds, colors, linewidths):
            u_min = self.get_lower_bound(lmbd)
            u_max = self.get_upper_bound(lmbd)
            ax = axs[0, 0]
            ax.add_patch(
                patches.Rectangle(
                    (u_min[0], u_min[1]),
                    u_max[0]-u_min[0],
                    u_max[1]-u_min[1],
                    alpha=alpha,
                    edgecolor=color,
                    facecolor="none",
                    linewidth=lw,
                ),
            )

            ax = axs[0, 1]
            ax.add_patch(
                patches.Rectangle(
                    (u_min[0], u_min[2]),
                    u_max[0]-u_min[0],
                    u_max[2]-u_min[2],
                    alpha=alpha,
                    edgecolor=color,
                    facecolor="none",
                    linewidth=lw,
                ),
            )

            ax = axs[1, 0]
            ax.add_patch(
                patches.Rectangle(
                    (u_min[0], u_min[3]),
                    u_max[0]-u_min[0],
                    u_max[3]-u_min[3],
                    alpha=alpha,
                    edgecolor=color,
                    facecolor="none",
                    linewidth=lw,
                ),
            )

            ax = axs[1, 1]
            ax.add_patch(
                patches.Rectangle(
                    (u_min[1], u_min[2]),
                    u_max[1]-u_min[1],
                    u_max[2]-u_min[2],
                    alpha=alpha,
                    edgecolor=color,
                    facecolor="none",
                    linewidth=lw,
                ),
            )

            ax = axs[2, 0]
            ax.add_patch(
                patches.Rectangle(
                    (u_min[1], u_min[3]),
                    u_max[1]-u_min[1],
                    u_max[3]-u_min[3],
                    alpha=alpha,
                    edgecolor=color,
                    facecolor="none",
                    linewidth=lw,
                ),
            )

            ax = axs[2, 1]
            ax.add_patch(
                patches.Rectangle(
                    (u_min[2], u_min[3]),
                    u_max[2]-u_min[2],
                    u_max[3]-u_min[3],
                    alpha=alpha,
                    edgecolor=color,
                    facecolor="none",
                    linewidth=lw,
                ),
            )
        return fig, axs
