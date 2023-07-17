import matplotlib.pyplot as plt
import numpy as np

from ftc.mission_determiners.polytope_determiner import (PolytopeDeterminer,
                                                         color_fader)
from ftc.models.multicopter import Multicopter


def test_polytope_determiner():
    print("Testing polytope_determiner...")
    multicopter = Multicopter(rtype="quad")
    B = multicopter.mixer.B
    u_min = multicopter.rotor_min * np.ones(B.shape[-1])
    u_max = multicopter.rotor_max * np.ones(B.shape[-1])
    #
    determiner = PolytopeDeterminer(u_min, u_max, lambda nu: np.linalg.pinv(B) @ nu)
    lmbd = np.array([1, 1, 1, 1])
    #
    nu_true = np.zeros(4)
    nu_false = 10 * np.array(
        [multicopter.m * multicopter.g, 0, 0, 0]
    )  # expected to be false
    for nu in [nu_true, nu_false]:
        is_in = determiner.determine_is_in(nu, lmbd)
        print(f"for generalized force {nu}, is_in = {is_in}")

    N = 10
    nus = np.linspace(nu_true, nu_false, N)
    lmbds = [lmbd for _ in range(N)]
    are_in = determiner.determine_are_in(nus, lmbds)
    print(are_in)


def test_draw():
    print("Testing draw...")
    N = 50
    multicopter = Multicopter(rtype="quad")
    B = multicopter.mixer.B
    u_min = multicopter.rotor_min * np.ones(B.shape[-1])
    u_max = multicopter.rotor_max * np.ones(B.shape[-1])
    #
    determiner = PolytopeDeterminer(u_min, u_max, lambda nu: np.linalg.pinv(B) @ nu)
    us = [u_max[0] * (i + 1) / N * np.ones(4) for i in range(N)]
    nus = [B @ u for u in us]
    lmbd = np.array([1, 1, 1, 1])
    lmbds = [
        ((N - (i + 1)) / N + (1 / 2)) * lmbd if i > N / 2 else lmbd for i in range(N)
    ]

    fig, axs = determiner.visualize(nus, lmbds)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_polytope_determiner()
    test_draw()
