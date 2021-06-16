import numpy as np


def calculate_recovery_rate(errors: np.ndarray, threshold: float=0.5):
    assert threshold > 0
    recovery_rate = np.average(np.abs(errors) <= threshold)
    return recovery_rate


if __name__ == "__main__":
    errors = np.array([
        [-1, 2],
        [2, 3],
        [0, 1],
    ])
    threshold = 1.0
    recovery_rate = calculate_recovery_rate(errors, threshold=threshold)
    print(recovery_rate)
