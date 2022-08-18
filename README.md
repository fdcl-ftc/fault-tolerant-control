# Flight Simulator for Fault-Tolerant-Control Algorithms

## Installation

```bash
git clone git@github.com:fdcl-ftc/fault-tolerant-control.git
cd fault-tolerant-control
pip install -e .
```

In Python 3.8>, use the module as follows.
```
import ftc
```

## Usage

### Register a Controller Class

The controller must be defined as a class in somewhere in a directory in `ftc.controllers`.

For example, to define a sliding mode controller, one must create a directory `ftc.controllers.mysmc`, and a module file `smc.py` in that directory.
The directory may contain several `.py` files for readability.
In any cases, the main controller class, let's say `MYSMC`, must be registered in `ftc.controllers.__init__.py` with an identifyable name (`id`) and a full entry point (`entry_point`) to that class as follows.

```python
register(
    id="My-SMC-v1",
    entry_point="ftc.controllers.mysmc.smc:MYSMC",
)
```

As you may notice, this registration process is borrowed from the OpenAI Gym module.

### Make a Controller Instance

After the controller is registered, one can make an instance of the controller class by `ftc.make` method.

```python
controller = ftc.make("My-SMC-v1")
```

## Controller Class API

### `get_control` Method

Each controller class must have the following `get_control` method which takes the current time and the main `env` object of the class `fym.BaseEnv` and returns the rotor force inputs in `(N, 1)` dimension `ndarray` along with some `dict`-type information.

```python
class MyController:
    def get_control(self, t, env):
        rfs = np.zeros((4, 1))
        info = {}
        return rfs, info
```

If the controller class also inherits `fym.BaseEnv` or `fym.BaseSystem`, the derivative must be defined in the `get_control` method.

```python
class MyController(fym.BaseEnv):
    def __init__(self):
        super().__init__()
        self.adaptive_part = fym.BaseSystem()

    def get_control(self, t, env):
        W = self.adaptive_part.state

        self.adaptive_part.dot = ...
```

## Env Class API

### `env.plant` Object

The passed `env` object to the `get_control` method has a `plant` object of class `fym.BaseEnv`, which is the main plant to be controlled.
Therefore, you may access the state vectors of the `plant`.

```python
    def get_control(self, t, env):
        pos, vel, R, omega = env.plant.observe_list()
        rfs = - pos ... 
```

### `env.get_ref` Method

Also, `env` object has a `get_ref` method which must be used in each controller class to get the reference profile.
The `get_ref` method takes several arguments.
The first argument is always the current time, and the subsequent arguments are the keys that you want to get.

```python
    def get_control(self, t, env):
        posd, posd_dot, posd_ddot = env.get_ref(t, "posd", "posd_dot", "posd_ddot")
```

### `env.get_Lambda` Method

For active fault tolerant control, one may require fault information of each rotor.
The `env` object provides `get_Lambda` method that takes the current time as an input and returns `Lambda` `(N, N)` dimensional ndarray which denotes the healthy of each rotor, i.e., `1 - LoE`.

```python
    def get_control(self, t, env):
        Lambda = env.get_Lambda(t)
```

Note that this `Lambda` is the estimated Lambda, and therefore if one wants a time-delayed estimation, the `get_Lambda` method must be modified.
For example, if `plant.get_Lambda` returns the true `Lambda` at the queried time, then the `env.get_Lambda` method should be

```python
class Env(fym.BaseEnv):
    def get_Lambda(self, t):
        return self.plant.get_Lambda(t - 0.02)
```
