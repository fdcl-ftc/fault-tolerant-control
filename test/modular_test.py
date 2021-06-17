import numpy as np
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time
import tqdm

import fym
from fym.utils.rot import angle2quat

from ftc.models.multicopter import Multicopter

cfg = fym.parser.parse({
    "parallel.max_workers": None,
    "episode.N": 100,
    "episode.range": {
        "pos": (-1, 1),
        "vel": (-1, 1),
        "omega": (-5, 5),
        "angle": (-5, 5),
    },
    "env.kwargs": {
        "dt": 0.01,
        "max_t": 10,
    },
    "ref.pos": np.vstack((0, 0, -10)),
    "path.run": Path("data", "run"),
})


class Env(fym.BaseEnv):
    def __init__(self, initial):
        super().__init__(**fym.parser.decode(cfg.env.kwargs))
        pos, vel, angle, omega = initial
        quat = angle2quat(*angle.ravel()[::-1])
        self.plant = Multicopter(pos, vel, quat, omega)

    def step(self):
        *_, done = self.update()
        return done

    def set_dot(self, t):
        rotors = np.zeros((self.plant.mixer.B.shape[1], 1))
        self.plant.set_dot(t, rotors)

        return dict(rotors=rotors, **self.observe_dict())


def single_run(i, initial):
    env = Env(initial)
    env.logger = fym.Logger(Path(cfg.path.run, f"env-{i:03d}.h5"))
    env.reset()

    while True:
        done = env.step()

        if done:
            break

    env.close()


def main():
    # Sampling initial conditions
    np.random.seed(0)
    pos = np.random.uniform(*cfg.episode.range.pos, size=(cfg.episode.N, 3, 1))
    vel = np.random.uniform(*cfg.episode.range.vel, size=(cfg.episode.N, 3, 1))
    angle = np.random.uniform(*cfg.episode.range.angle, size=(cfg.episode.N, 3, 1))
    omega = np.random.uniform(*cfg.episode.range.omega, size=(cfg.episode.N, 3, 1))
    initial_set = np.stack((pos, vel, angle, omega), axis=1)

    # Initialize concurrent
    cpu_workers = os.cpu_count()
    max_workers = int(cfg.parallel.max_workers or cpu_workers)
    assert max_workers <= os.cpu_count(), \
        f"workers should be less than {cpu_workers}"
    print(f"Sample with {max_workers} workers ...")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers) as p:
        list(tqdm.tqdm(
            p.map(single_run, range(cfg.episode.N), initial_set),
            total=cfg.episode.N
        ))

    print(f"Elapsed time is {time.time() - t0:5.2f} seconds."
          f" > data saved in \"{cfg.path.run}\"")


if __name__ == "__main__":
    main()
