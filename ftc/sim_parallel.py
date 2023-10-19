import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fym
import numpy as np
import tqdm


def sim(i, initial, Env):
    dirpath = "data"
    loggerpath = Path(dirpath, f"env_{i:04d}.h5")
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
            alt_mae=calculate_mae(data, env.time_from, error_type="alt"),
            eval_mfa=evaluate_mfa(data, env.time_from, verbose=True),
        ),
    )


def sim_parallel(N, initials, Env, workers=None):
    cpu_workers = os.cpu_count()
    workers = int(workers or cpu_workers)
    assert workers <= os.cpu_count(), f"workers should be less than {cpu_workers}"
    print(f"Sample with {workers} workers ...")
    with ProcessPoolExecutor(workers) as p:
        list(tqdm.tqdm(p.map(sim, range(N), initials, itertools.repeat(Env)), total=N))


def get_errors(data, time_from=5, error_type=None):
    time_index = data["env"]["t"] > max(data["env"]["t"]) - time_from
    if error_type == "alt":
        errors = (
            data["env"]["posd"][time_index, 2, 0]
            - data["env"]["plant"]["pos"][time_index, 2, 0]
        )
    else:
        errors = (
            data["env"]["posd"][time_index, :, 0]
            - data["env"]["plant"]["pos"][time_index, :, 0]
        ).squeeze()
    return errors


def calculate_mae(data, time_from=5, error_type=None):
    errors = get_errors(data, time_from, error_type)
    if bool(list(errors)):
        mae = np.mean(np.abs(errors), axis=0)
    else:
        mae = []
    return mae


def calculate_recovery_rate(errors, threshold=0.5):
    assert threshold > 0
    if bool(list(errors)):
        recovery_rate = np.mean(np.abs(errors) <= threshold)
    else:
        recovery_rate = 0
    return recovery_rate


def evaluate_recovery_rate(N, threshold=0.5, dirpath="data"):
    alt_maes = []
    for i in range(N):
        _, info = fym.load(Path(dirpath, f"env_{i:04d}.h5"), with_info=True)
        alt_maes = np.append(alt_maes, info["alt_mae"])
    recovery_rate = calculate_recovery_rate(alt_maes, threshold=threshold)
    print(f"Recovery rate is {recovery_rate:.3f}.")


def evaluate_mfa(data, time_from=5, threshold=0.5 * np.ones(3), verbose=False):
    """
    Is the mission feasibility assessment success?
    """
    mae = calculate_mae(data, time_from)
    eval = np.all(mae <= threshold)

    mfa = np.all(data["env"]["mfa"])
    if verbose:
        print(f"MAE of position trajectory is {mae}.")
        if mfa == eval:
            print(f"MFA Success: MFA={mfa}, evaluation={eval}")
        else:
            print(f"MFA Fails: MFA={mfa}, evaluation={eval}")
    return mfa == eval


def evaluate_mfa_success_rate(N, dirpath="data"):
    eval_mfas = []
    for i in range(N):
        _, info = fym.load(Path(dirpath, f"env_{i:04d}.h5"), with_info=True)
        eval_mfas = np.append(eval_mfas, info["eval_mfa"])
    mfa_success_rate = np.mean(eval_mfas)
    print(f"MFA rate is {mfa_success_rate:.3f}.")
