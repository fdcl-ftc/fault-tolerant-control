import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
import fire
matplotlib.rc('font', **{"size": 15, },)


def read_file(path):
    traj = {}
    with h5py.File(path, "r") as f:
        traj["plant"] = {
            "pos": np.array(f["env"]["plant"]["pos"]),
            "vel": np.array(f["env"]["plant"]["vel"]),
            "omega": np.array(f["env"]["plant"]["omega"]),
        }
        traj["ang"] = np.array(f["env"]["ang"])
        traj["ctrls"] = np.array(f["env"]["ctrls"])
        traj["t"] = np.array(f["env"]["t"])
        traj["posd"] = np.array(f["env"]["posd"])
        traj["angd"] = np.array(f["env"]["angd"])
    return traj


def main(path, n, legend=False):
    """
    The original author: Hanna Lee, SNU
    Modified by Jinrae Kim, SNU
    """
    print(f"case number: {n}")
    dataopt = read_file(f"{path}/plse/env_{n:05}.h5")
    datatest = read_file(f"{path}/test/env_{n:05}.h5")  # random = test
    datafixed = read_file(f"{path}/fixed/env_{n:05}.h5")
    datafnn = read_file(f"{path}/fnn/env_{n:05}.h5")

    """ Notes """
    lsopt = "r-"
    lstest = "b-."
    lsfixed = "g:"
    lsfnn = "c-."
    lsref = "k-"

    """ Figure 1 - States """
    fig, axes = plt.subplots(2, 4, figsize=(18, 5), squeeze=False, sharex=True)

    """ Column 1 """
    ax = axes[0, 0]
    ax.plot(datatest["t"], datatest["plant"]["pos"][:, 2].squeeze(-1), lstest)
    ax.plot(datafixed["t"], datafixed["plant"]["pos"][:, 2].squeeze(-1), lsfixed)
    ax.plot(datafnn["t"], datafnn["plant"]["pos"][:, 2].squeeze(-1), lsfnn)
    ax.plot(dataopt["t"], dataopt["plant"]["pos"][:, 2].squeeze(-1), lsopt)
    ax.plot(dataopt["t"], dataopt["posd"][:, 2].squeeze(-1), lsref)
    ax.set_ylabel(r"$z$, m")

    if legend:
        ax.legend(["random", "fixed", "FNN", "PLSE"], loc="upper right")

    ax.set_xlabel("Time, sec")

    ax = axes[1, 0]
    ax.plot(datatest["t"], datatest["plant"]["vel"][:, 2].squeeze(-1), lstest)
    ax.plot(datafixed["t"], datafixed["plant"]["vel"][:, 2].squeeze(-1), lsfixed)
    ax.plot(datafnn["t"], datafnn["plant"]["vel"][:, 2].squeeze(-1), lsfnn)
    ax.plot(dataopt["t"], dataopt["plant"]["vel"][:, 2].squeeze(-1), lsopt)
    ax.plot(dataopt["t"], dataopt["posd"][:, 2].squeeze(-1), lsref)
    ax.set_ylabel(r"$v_z$, m/s")

    ax.set_xlabel("Time, sec")

    ax = axes[0, 1]
    ax.plot(datatest["t"], np.rad2deg(datatest["ang"][:, 0].squeeze(-1)), lstest)
    ax.plot(datafixed["t"], np.rad2deg(datafixed["ang"][:, 0].squeeze(-1)), lsfixed)
    ax.plot(datafnn["t"], np.rad2deg(datafnn["ang"][:, 0].squeeze(-1)), lsfnn)
    ax.plot(dataopt["t"], np.rad2deg(dataopt["ang"][:, 0].squeeze(-1)), lsopt)
    ax.plot(dataopt["t"], np.rad2deg(dataopt["angd"][:, 0].squeeze(-1)), lsref)
    ax.set_ylabel(r"$\phi$, deg")

    ax = axes[0, 2]
    ax.plot(datatest["t"], np.rad2deg(datatest["ang"][:, 1].squeeze(-1)), lstest)
    ax.plot(datafixed["t"], np.rad2deg(datafixed["ang"][:, 1].squeeze(-1)), lsfixed)
    ax.plot(datafnn["t"], np.rad2deg(datafnn["ang"][:, 1].squeeze(-1)), lsfnn)
    ax.plot(dataopt["t"], np.rad2deg(dataopt["ang"][:, 1].squeeze(-1)), lsopt)
    ax.plot(dataopt["t"], np.rad2deg(dataopt["angd"][:, 1].squeeze(-1)), lsref)
    ax.set_ylabel(r"$\theta$, deg")

    """ Column 2 """
    ax = axes[0, 3]
    ax.plot(datatest["t"], np.rad2deg(datatest["ang"][:, 2].squeeze(-1)), lstest)
    ax.plot(datafixed["t"], np.rad2deg(datafixed["ang"][:, 2].squeeze(-1)), lsfixed)
    ax.plot(datafnn["t"], np.rad2deg(datafnn["ang"][:, 2].squeeze(-1)), lsfnn)
    ax.plot(dataopt["t"], np.rad2deg(dataopt["ang"][:, 2].squeeze(-1)), lsopt)
    ax.plot(dataopt["t"], np.rad2deg(dataopt["angd"][:, 2].squeeze(-1)), lsref)
    ax.set_ylabel(r"$\psi$, deg")

    ax.set_xlabel("Time, sec")

    ax = axes[1, 1]
    ax.plot(datatest["t"], np.rad2deg(datatest["plant"]["omega"][:, 0].squeeze(-1)), lstest)
    ax.plot(datafixed["t"], np.rad2deg(datafixed["plant"]["omega"][:, 0].squeeze(-1)), lsfixed)
    ax.plot(datafnn["t"], np.rad2deg(datafnn["plant"]["omega"][:, 0].squeeze(-1)), lsfnn)
    ax.plot(dataopt["t"], np.rad2deg(dataopt["plant"]["omega"][:, 0].squeeze(-1)), lsopt)
    ax.set_ylabel(r"$p$, deg/s")

    ax = axes[1, 2]
    ax.plot(datatest["t"], np.rad2deg(datatest["plant"]["omega"][:, 1].squeeze(-1)), lstest)
    ax.plot(datafixed["t"], np.rad2deg(datafixed["plant"]["omega"][:, 1].squeeze(-1)), lsfixed)
    ax.plot(datafnn["t"], np.rad2deg(datafnn["plant"]["omega"][:, 1].squeeze(-1)), lsfnn)
    ax.plot(dataopt["t"], np.rad2deg(dataopt["plant"]["omega"][:, 1].squeeze(-1)), lsopt)
    ax.set_ylabel(r"$q$, deg/s")

    ax = axes[1, 3]
    ax.plot(datatest["t"], np.rad2deg(datatest["plant"]["omega"][:, 2].squeeze(-1)), lstest)
    ax.plot(datafixed["t"], np.rad2deg(datafixed["plant"]["omega"][:, 2].squeeze(-1)), lsfixed)
    ax.plot(datafnn["t"], np.rad2deg(datafnn["plant"]["omega"][:, 2].squeeze(-1)), lsfnn)
    ax.plot(dataopt["t"], np.rad2deg(dataopt["plant"]["omega"][:, 2].squeeze(-1)), lsopt)
    ax.set_ylabel(r"$r$, deg/s")

    ax.set_xlabel("Time, sec")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    fig.align_ylabels(axes)

    # """ Figure 2 - Rotor thrusts """
    # fig, axs = plt.subplots(3, 2, sharex=True)
    # ylabels = np.array((["Rotor 1", "Rotor 2"],
    #                     ["Rotor 3", "Rotor 4"],
    #                     ["Rotor 5", "Rotor 6"]))
    # for i, _ylabel in np.ndenumerate(ylabels):
    #     x, y = i
    #     ax = axs[i]
    #     ax.plot(dataopt["t"], dataopt["ctrls"].squeeze(-1)[:, 2*x+y], lsopt, label="PLSE")
    #     ax.plot(datatest["t"], datatest["ctrls"].squeeze(-1)[:, 2*x+y], lstest, label="random")
    #     ax.plot(datafix2["t"], datafix2["ctrls"].squeeze(-1)[:, 2*x+y], lsfix2, label="fixed (type a)")
    #     ax.plot(datafix3["t"], datafix3["ctrls"].squeeze(-1)[:, 2*x+y], lsfix3, label="fixed (type b)")
    #     ax.plot(datafnn["t"], datafnn["ctrls"].squeeze(-1)[:, 2*x+y], lsfnn, label="FNN")
    #     # ax.plot(data["t"], dataopt["ctrls"].squeeze(-1)[:, 2*x+y], lsopt, label="PLSE")
    #     # ax.plot(data["t"], datatest["ctrls"].squeeze(-1)[:, 2*x+y], lstest, label="random")
    #     # ax.plot(data["t"], datafix2["ctrls"].squeeze(-1)[:, 2*x+y], lsfix2, label="fixed (type a)")
    #     # ax.plot(data["t"], datafix3["ctrls"].squeeze(-1)[:, 2*x+y], lsfix3, label="fixed (type b)")
    #     # ax.plot(data["t"], datafnn["ctrls"].squeeze(-1)[:, 2*x+y], lsfnn, label="FNN")
    #     # ax.plot(data["t"], data["ctrls0"].squeeze(-1)[:, 2*x+y], lsref, label="ref")
    #     ax.grid()
    #     if i == (0, 1):
    #         ax.legend(loc="upper right")
    #     plt.setp(ax, ylabel=_ylabel)
    #     ax.set_ylim([1000-5, 2000+5])
    # plt.gcf().supxlabel("Time, sec")
    # plt.gcf().supylabel("Rotor Thrusts")
    #
    # fig.tight_layout()
    # fig.subplots_adjust(wspace=0.5)
    # fig.align_ylabels(axs)

    # plt.show()
    plt.savefig(f"traj_{n:05}.pdf")


if __name__ == "__main__":
    fire.Fire(main)
