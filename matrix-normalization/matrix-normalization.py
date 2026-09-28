import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    M = np.asarray(matrix, dtype=float)

    if norm_type == "l1":
        norms = np.sum(np.abs(M), axis=axis, keepdims=True)
    elif norm_type == "l2":
        norms = np.sqrt(np.sum(M**2, axis=axis, keepdims=True))
    else:
        norms = np.max(np.abs(M), axis=axis, keepdims=True)

    return M / np.where(norms == 0.0, 1.0, norms)