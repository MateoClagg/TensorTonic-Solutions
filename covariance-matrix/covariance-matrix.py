import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    X = np.asarray(X, dtype=float)
    N = len(X)
    Xc = X - np.mean(X, axis=0)

    return (Xc.T @ Xc) / (len(X) - 1) 