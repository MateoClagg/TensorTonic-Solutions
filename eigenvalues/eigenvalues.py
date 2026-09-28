import numpy as np

def calculate_eigenvalues(matrix: list) -> np.ndarray:
    """
    Returns a sorted NumPy array of real eigenvalues.
    """
    M = np.asarray(matrix, dtype=float)

    eig_vals = np.linalg.eigvals(M)

    return np.sort(eig_vals)