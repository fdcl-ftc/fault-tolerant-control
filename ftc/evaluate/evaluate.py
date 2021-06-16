import numpy as np


def calculate_recovery_rate(errors: np.ndarray, threshold: float=0.5):
    assert threshold > 0
    is_success_array = np.zeros_like(errors)
    for i, error in enumerate(errors):
        is_success_array[i] = abs(error) <= threshold  # the absolute value of error
    recovery_rate = np.sum(is_success_array) / np.prod(is_success_array.shape)
    return recovery_rate


if __name__ == "__main__":
    errors = np.array([
        [1, 2],
        [2, 3],
        [0, 1],
    ])
    threshold = 1.0
    recovery_rate = calculate_recovery_rate(errors, threshold=threshold)
    print(recovery_rate)
