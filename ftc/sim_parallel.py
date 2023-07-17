import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fym
import numpy as np
import tqdm


def sim(i, initial, Env):
    loggerpath = Path("data", f"env_{i:04d}.h5")
    env = Env(initial)
    flogger = fym.Logger(loggerpath)

    env.reset()

    while True:
        env.render(mode=None)

        done, env_info = env.step()
        flogger.record(env=env_info, initial=initial)

        if done:
            break

    flogger.close()

    data = fym.load(loggerpath)
    time_index = data["env"]["t"] > env.tf - env.cuttime
    alt_error = (
        data["env"]["posd"][time_index, 2, 0]
        - data["env"]["plant"]["pos"][time_index, 2, 0]
    )
    if bool(list(alt_error)):
        mae = np.mean(alt_error)
    else:
        mae = []
    fym.save(loggerpath, data, info=dict(alt_error=mae))


def sim_parallel(N, initials, Env, workers=None):
    cpu_workers = os.cpu_count()
    workers = int(workers or cpu_workers)
    assert workers <= os.cpu_count(), f"workers should be less than {cpu_workers}"
    print(f"Sample with {workers} workers ...")
    with ProcessPoolExecutor(workers) as p:
        list(tqdm.tqdm(p.map(sim, range(N), initials, itertools.repeat(Env)), total=N))


def calculate_recovery_rate(errors, threshold=0.5):
    assert threshold > 0
    if bool(list(errors)):
        recovery_rate = np.average(np.abs(errors) <= threshold)
    else:
        recovery_rate = 0
    return recovery_rate


def evaluate(N, threshold=0.5):
    alt_errors = []
    for i in range(N):
        _, info = fym.load(Path("data", f"env_{i:04d}.h5"), with_info=True)
        alt_errors = np.append(alt_errors, info["alt_error"])
    recovery_rate = calculate_recovery_rate(alt_errors, threshold=threshold)
    print(f"Recovery rate is {recovery_rate:.3f}.")


def evaluate_pos(threshold=np.ones(3)):
    data = fym.load("data.h5")["env"]
    errors = (data["posd"] - data["plant"]["pos"]).squeeze()
    error_norms = np.linalg.norm(errors, axis=0)
    print(f"Position trajectory error norms are {error_norms}.")
    return np.all(error_norms <= threshold)


def evaluate_mfa(eval, verbose=False):
    """
    Is the mission feasibility assessment success?
    """
    data = fym.load("data.h5")["env"]
    mfa = np.all(data["mfa"])
    if mfa == eval:
        if verbose:
            print("MFA Success")
    else:
        if verbose:
            print("MFA Fails")
