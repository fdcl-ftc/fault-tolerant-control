import numpy as np


class PolytopeDeterminer:
    """
        PolytopeDeterminer

    A mission-success determiner using polytope.
    """

    def __init__(
        self,
        u_min,
        u_max,
        allocator,
    ):
        """
        u_min: (m,) array; minimum input (element-wise)
        u_max: (m,) array; maximum input (element-wise)
        allocator: A control allocation law supposed to be used in a controller of interest

        Here, input space `U` is defined as `u_min <= u <= u_max` (element-wise).
        """
        self.u_min = u_min
        self.u_max = u_max
        self.allocator = allocator

    def determine_is_in(self, nu, lmbd, scaling_factor: float):
        """
            determine_is_in

        Determine if the control input (will be allocated) is in input space U for given generalized force `nu` and actuator fault information `lmbd`.

        nu = [F, M^T]^T: (4,) array; generalized force
        B: (4 x m) array; control effectiveness matrix
        lmbd: (m,) array containing actuator fault information, ranging from 0 to 1 (effectiveness).
        """
        assert scaling_factor >= 0.0
        u = self.allocator(nu)
        is_larger_than_min = u >= scaling_factor * np.diag(lmbd) @ self.u_min
        is_smaller_than_max = u <= scaling_factor * np.diag(lmbd) @ self.u_max
        is_in = np.all(is_larger_than_min & is_smaller_than_max)
        return is_in

    def determine_are_in(self, nus, lmbds, scaling_factor=1.0):
        """
        Apply `determine_is_in` for multiple pairs of (nu, lmbd)'s.
        """
        are_in = [
            self.determine_is_in(nu, lmbd, scaling_factor)
            for (nu, lmbd) in zip(nus, lmbds)
        ]
        return are_in

    def determine(self, *args, **kwargs):
        """
        Is the mission success? (by considering all points along a trajectory imply that corresponding contrl inputs are in the input space.)
        """
        are_in = self.determine_are_in(*args, **kwargs)
        return np.all(are_in)
