import numpy as np

def t_test_one_sample(x: list, mu0: float) -> float:
    """
    Returns the t-statistic as a float.
    """
    x = np.asarray(x, dtype=float)

    s = np.sqrt(np.sum((x - np.mean(x))**2) / (len(x) - 1) )

    if s == 0:
        if np.mean(x) == mu0:
            return 0
        else:
            return np.inf

    return ( ( np.mean(x) - mu0 ) / (s / np.sqrt(len(x)) ) ).item()