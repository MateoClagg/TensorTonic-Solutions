import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    x = np.asarray(x, dtype=float)

    pmf = (1-p)**(1-x) * (p)**(x)

    return {
        "pmf": pmf,
        "mean": float(p),
        "variance": float(p*(1-p))
    }