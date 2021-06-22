import numpy as np
import matplotlib.pyplot as plt
import fym
from fym.utils.rot import angle2quat, quat2angle


def exp_plot(loggerpath):
    data, info = fym.load(loggerpath, with_info=True)
    rotor_min = info["rotor_min"]
    rotor_max = info["rotor_max"]

    # FDI
    plt.figure()

    ax = plt.subplot(321)
    for i in range(data["W"].shape[1]):
        if i != 0:
            plt.subplot(321+i, sharex=ax)
        plt.ylim([0-0.1, 1+0.1])
        plt.plot(data["t"], data["W"][:, i, i], "r--", label="Actual")
        plt.plot(data["t"], data["What"][:, i, i], "k-", label="Estimated")
        if i == 0:
            plt.legend()
    plt.gcf().supylabel("FDI")
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()

    # Rotor
    plt.figure()

    ax = plt.subplot(321)
    for i in range(data["rotors"].shape[1]):
        if i != 0:
            plt.subplot(321+i, sharex=ax)
        plt.ylim([rotor_min-5, rotor_max+5])
        plt.plot(data["t"], data["rotors"][:, i], "k-", label="Response")
        plt.plot(data["t"], data["rotors_cmd"][:, i], "r--", label="Command")
        if i == 0:
            plt.legend()
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Rotor thrust")
    plt.tight_layout()

    # Position
    plt.figure()
    plt.ylim([-5, 5])

    for i, (_label, _ls) in enumerate(zip(["x", "y", "z"], ["-", "--", "-."])):
        plt.plot(data["t"], data["x"]["pos"][:, i, 0], "k"+_ls, label=_label)
        plt.plot(data["t"], data["ref"][:, i, 0], "r"+_ls, label=_label+" (cmd)")
    # plt.axvspan(3, 3.042, alpha=0.2, color="b")
    # plt.axvline(3.042, alpha=0.8, color="b", linewidth=0.5)

    # plt.axvspan(6, 6.011, alpha=0.2, color="b")
    # plt.axvline(6.011, alpha=0.8, color="b", linewidth=0.5)

    # plt.annotate("Rotor 0 fails", xy=(3, 0), xytext=(3.5, 0.5),
    #              arrowprops=dict(arrowstyle='->', lw=1.5))
    # plt.annotate("Rotor 2 fails", xy=(6, 0), xytext=(7.5, 0.2),
    #              arrowprops=dict(arrowstyle='->', lw=1.5))
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Position, m")
    plt.tight_layout()
    plt.legend()

    # velocity
    plt.figure()
    plt.ylim([-5, 5])

    for i, (_label, _ls) in enumerate(zip(["Vx", "Vy", "Vz"], ["-", "--", "-."])):
        plt.plot(data["t"], data["x"]["vel"][:, i, 0], "k"+_ls, label=_label)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Velocity, m/s")
    plt.tight_layout()
    plt.legend()

    # euler angles
    plt.figure()
    plt.ylim([-40, 40])

    angles = np.vstack([quat2angle(data["x"]["quat"][j, :, 0]) for j in range(len(data["x"]["quat"][:, 0, 0]))])
    for i, (_label, _ls) in enumerate(zip(["yaw", "pitch", "roll"], ["-.", "--", "-"])):
        plt.plot(data["t"], np.rad2deg(angles[:, i]), "k"+_ls, label=_label)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Euler angles, deg")
    plt.tight_layout()
    plt.legend()

    # angular rates
    plt.figure()
    plt.ylim([-90, 90])

    for i, (_label, _ls) in enumerate(zip(["p", "q", "r"], ["-.", "--", "-"])):
        plt.plot(data["t"], np.rad2deg(data["x"]["omega"][:, i, 0]), "k"+_ls, label=_label)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Angular rates, deg/s")
    plt.tight_layout()
    plt.legend()

    plt.show()
