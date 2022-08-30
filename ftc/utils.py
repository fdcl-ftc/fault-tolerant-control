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
