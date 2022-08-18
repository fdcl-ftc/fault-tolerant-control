from ftc.registration import registry


def make(id):
    return registry[id]()


def get_controllers(*args):
    controllers = []
    for id in args:
        controllers.append(make(id))

    return controllers or [Controller() for Controller in registry.values()]
