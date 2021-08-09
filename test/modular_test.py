import numpy as np
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time
import tqdm

import fym
from fym.utils.rot import angle2quat

import ftc.config
from ftc.models.multicopter import Multicopter
from ftc.faults.actuator import LoE
from ftc.faults.manager import LoEManager
from ftc.evaluate.evaluate import calculate_recovery_rate
from ftc.agents.CA import ConstrainedCA
import ftc.agents.lqr as lqr
from ftc.plotting import exp_plot

cfg = ftc.config.load()
ftc.config.set({
    "path.run": Path("data", "run"),
})


class Env(fym.BaseEnv):
    def __init__(self, initial):
        super().__init__(**fym.parser.decode(cfg.env.kwargs))
        pos, vel, angle, omega = initial
        pos[2] = pos[2] - 10
        quat = angle2quat(*angle.ravel()[::-1])
        self.plant = Multicopter(pos, vel, quat, omega)
        self.trim_forces = np.vstack([self.plant.m * self.plant.g, 0, 0, 0])
        n = self.plant.mixer.B.shape[1]

        # Define faults
        self.sensor_faults = []
        self.fault_manager = LoEManager([
            LoE(time=3, index=0, level=0.),  # scenario a
            # LoE(time=6, index=2, level=0.),  # scenario b
        ], no_act=n)

        # Define FDI
        self.fdi = self.fault_manager.fdi

        # Define agents
        self.CCA = ConstrainedCA(self.plant.mixer.B)
        self.controller = lqr.LQRController(self.plant.Jinv,
                                            self.plant.m,
                                            self.plant.g)

        self.detection_time = self.fault_manager.fault_times + self.fdi.delay

    def step(self):
        *_, done = self.update()
        return done

    def control_allocation(self, t, forces, What):
        fault_index = self.fdi.get_index(t)

        if len(fault_index) == 0:
            rotors = np.linalg.pinv(self.plant.mixer.B.dot(What)).dot(forces)
        else:
            rotors = self.CCA.solve_opt(fault_index, forces,
                                        self.plant.rotor_min,
                                        self.plant.rotor_max)
        return rotors

    def get_ref(self, t):
        pos_des = cfg.ref.pos
        vel_des = np.vstack([0, 0, 0])
        quat_des = np.vstack([1, 0, 0, 0])
        omega_des = np.vstack([0, 0, 0])
        ref = np.vstack([pos_des, vel_des, quat_des, omega_des])

        return ref

    def set_dot(self, t):
        mult_states = self.plant.state
        W = self.fdi.get_true(t)
        What = self.fdi.get(t)
        ref = self.get_ref(t)

        # Controller
        forces = self.controller.get_FM(mult_states, ref)

        rotors_cmd = self.control_allocation(t, forces, What)

        # actuator saturation
        rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)

        # Set actuator faults
        rotors = self.fault_manager.get_faulty_input(t, rotors)

        self.plant.set_dot(t, rotors)

        return dict(t=t, x=self.plant.observe_dict(), What=What,
                    rotors=rotors, rotors_cmd=rotors_cmd, W=W, ref=ref)


def single_run(i, initial):
    loggerpath = Path(cfg.path.run, f"env-{i:03d}.h5")
    env = Env(initial)
    env.logger = fym.Logger(loggerpath)
    env.reset()

    while True:
        done = env.step()

        if done:
            env_info = {
                "detection_time": env.detection_time,
                "rotor_min": env.plant.rotor_min,
                "rotor_max": env.plant.rotor_max,
            }
            env.logger.set_info(**env_info)
            break

    env.close()

    data, info = fym.load(loggerpath, with_info=True)
    time_index = data["t"] > cfg.env.kwargs.max_t - cfg.evaluation.cuttime
    alt_error = cfg.ref.pos[2] - data["x"]["pos"][time_index, 2, 0]
    fym.parser.update(info, dict(alt_error=np.mean(alt_error)))
    fym.save(loggerpath, data, info=info)


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
    alt_errors = []
    for i in range(cfg.episode.N):
        loggerpath = Path(cfg.path.run, f"env-{i:03d}.h5")
        data, info = fym.load(loggerpath, with_info=True)
        alt_errors = np.append(alt_errors, info["alt_error"])
    recovery_rate = calculate_recovery_rate(alt_errors, threshold=0.5)
    print(f"Recovery rate is {recovery_rate:.3f}.")
    exp_plot(loggerpath)
