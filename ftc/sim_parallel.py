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
    fym.save(
        loggerpath,
        data,
        info=dict(
            mae=calculate_mae(data, env.cuttime),
            # eval_mfa=evaluate_mfa(data, evaluate_pos_error(data, env.cuttime)),
        ),
    )


def sim_parallel(N, initials, Env, workers=None):
    cpu_workers = os.cpu_count()
    workers = int(workers or cpu_workers)
    assert workers <= os.cpu_count(), f"workers should be less than {cpu_workers}"
    print(f"Sample with {workers} workers ...")
    with ProcessPoolExecutor(workers) as p:
        list(tqdm.tqdm(p.map(sim, range(N), initials, itertools.repeat(Env)), total=N))


def calculate_mae(data, cuttime=5):
    time_index = data["env"]["t"] > max(data["env"]["t"]) - cuttime
    alt_error = (
        data["env"]["posd"][time_index, 2, 0]
        - data["env"]["plant"]["pos"][time_index, 2, 0]
    )
    if bool(list(alt_error)):
        mae = np.mean(alt_error)
    else:
        mae = []
    return mae


def calculate_recovery_rate(errors, threshold=0.5):
    assert threshold > 0
    if bool(list(errors)):
        recovery_rate = np.average(np.abs(errors) <= threshold)
    else:
        recovery_rate = 0
    return recovery_rate


def evaluate_recovery_rate(N, threshold=0.5):
    maes = []
    for i in range(N):
        _, info = fym.load(Path("data", f"env_{i:04d}.h5"), with_info=True)
        maes = np.append(maes, info["mae"])
    recovery_rate = calculate_recovery_rate(maes, threshold=threshold)
    print(f"Recovery rate is {recovery_rate:.3f}.")


def evaluate_pos_error(data, cuttime=5, threshold=np.ones(3)):
    time_index = data["env"]["t"] > max(data["env"]["t"]) - cuttime
    errors = (
        data["posd"][time_index, :, 0] - data["env"]["plant"]["pos"][time_index, :, 0]
    ).squeeze()
    error_norms = np.linalg.norm(errors, axis=0)
    print(f"Position trajectory error norms are {error_norms}.")
    return np.all(error_norms <= threshold)


def evaluate_mfa(data, eval, verbose=False):
    """
    Is the mission feasibility assessment success?
    """
    mfa = np.all(data["env"]["mfa"])
    if mfa == eval:
        if verbose:
            print("MFA Success")
    else:
        if verbose:
            print("MFA Fails")


def evaluate_mfa_success_rate(N):
    eval_mfas = []
    for i in range(N):
        _, info = fym.load(Path("data", f"env_{i:04d}.h5"), with_info=True)
        eval_mfas = np.append(eval_mfas, info["eval_mfa"])
    mfa_success_rate = np.average(eval_mfas)
    print(f"MFA rate is {mfa_success_rate:.3f}.")
