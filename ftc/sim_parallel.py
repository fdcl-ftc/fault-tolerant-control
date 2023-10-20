import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fym
import numpy as np
import tqdm


def sim_parallel(sim, N, initials, Env, workers=None):
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


def evaluate_recovery_rate(
    N, time_from=5, error_type="alt", threshold=0.5, dirpath="data"
):
    alt_maes = []
    for i in range(N):
        data = fym.load(Path(dirpath, f"env_{i:04d}.h5"))
        alt_mae = calculate_mae(data, time_from=time_from, error_type=error_type)
        alt_maes = np.append(alt_maes, alt_mae)
    recovery_rate = calculate_recovery_rate(alt_maes, threshold=threshold)
    print(f"Recovery rate is {recovery_rate:.3f}.")


def evaluate_mfa(mfa, mae, threshold=0.5 * np.ones(3), verbose=False):
    """
    Is the mission feasibility assessment success?
    """
    eval = np.all(mae <= threshold)
    if verbose:
        print(f"MAE of position trajectory is {mae}.")
        if mfa == eval:
            print(f"MFA Success: MFA={mfa}, evaluation={eval}")
        else:
            print(f"MFA Fails: MFA={mfa}, evaluation={eval}")
    return mfa == eval


def evaluate_mfa_success_rate(
    N,
    time_from=5,
    error_type=None,
    threshold=0.5 * np.ones(3),
    dirpath="data",
    verbose=False,
):
    evals = []
    for i in range(N):
        data = fym.load(Path(dirpath, f"env_{i:04d}.h5"))
        mfa = np.all(data["env"]["mfa"])
        mae = calculate_mae(data, time_from=time_from, error_type=error_type)
        evals = np.append(
            evals, evaluate_mfa(mfa, mae, threshold=threshold, verbose=verbose)
        )
    mfa_success_rate = np.mean(evals)
    print(f"MFA rate is {mfa_success_rate:.3f}.")
