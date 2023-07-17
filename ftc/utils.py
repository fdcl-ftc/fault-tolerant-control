from copy import deepcopy
from functools import reduce

import numpy as np

import fym
from ftc.registration import registry


def make(id, env=None):
    assert env is not None
    return registry[id](env)


def get_controllers(*args, env=None):
    assert env is not None
    controllers = []
    for id in args:
        controllers.append(make(id, env=env))

    return controllers or [Controller() for Controller in registry.values()]


def safeupdate(*configs):
    assert len(configs) > 1

    def _merge(base, new):
        assert isinstance(base, dict), f"{base} is not a dict"
        assert isinstance(new, dict), f"{new} is not a dict"
        out = deepcopy(base)
        for k, v in new.items():
            # assert k in out, f"{k} not in {base}"
            if isinstance(v, dict):
                if "grid_search" in v:
                    out[k] = v
                else:
                    out[k] = _merge(out[k], v)
            else:
                out[k] = v

        return out

    return reduce(_merge, configs)


def linearization(statefunc, states, ctrls, ptrb):
    """
    Parameters
    ------------
    statefunc : callable function that returns nx1 states derivatives vector
    states : nx1 vector
    ctrls : mx1 vector
    ptrb : numerical ptrb size

    Return
    -------------
    linearized matrix A, B for peturbed states and ctrls
    dxdot = Adx + Bdu
    """

    n = np.size(states)
    m = np.size(ctrls)
    A = np.zeros((n, n))
    B = np.zeros((n, m))

    for i in np.arange(n):
        ptrbvec_x = np.zeros((n, 1))
        ptrbvec_x[i] = ptrb
        x_ptrb = states + ptrbvec_x

        dfdx = (statefunc(x_ptrb, ctrls) - statefunc(states, ctrls)) / ptrb
        for j in np.arange(n):
            A[j, i] = dfdx[j]

    for i in np.arange(m):
        ptrbvec_u = np.zeros((m, 1))
        ptrbvec_u[i] = ptrb
        u_ptrb = ctrls + ptrbvec_u

        dfdu = (statefunc(states, u_ptrb) - statefunc(states, ctrls)) / ptrb
        for j in np.arange(n):
            B[j, i] = dfdu[j]

    return A, B


def evaluate_pos(threshold=np.ones(3)):
    data = fym.load("data.h5")["env"]
    errors = (data["posd"] - data["plant"]["pos"]).squeeze()
    error_norms = np.linalg.norm(errors, axis=0)
    print(f"Position trajectory error norms are {error_norms}.")
    return np.all(error_norms <= threshold)


def evaluate_mfa(eval, verbose=False):
    data = fym.load("data.h5")["env"]
    mfa = np.all(data["mfa"])
    if mfa == eval:
        if verbose:
            print("MFA Success")
    else:
        if verbose:
            print("MFA Fails")
