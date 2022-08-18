import importlib


registry = {}


def register(id: str, entry_point: str):
    module, attr = entry_point.split(":", maxsplit=1)
    mod = importlib.import_module(module)
    fn = getattr(mod, attr)
    registry[id] = fn
