from copy import deepcopy
from functools import reduce

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
