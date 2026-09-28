import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    M = np.asarray(matrix, dtype=float)

    if norm_type == "l1":
        if axis == None: 
            l1_norm = np.sum(np.abs(M))
        elif axis == 0:
            l1_norm = np.sum(np.abs(M), axis=0, keepdims=True)
        else:
            l1_norm = np.sum(np.abs(M), axis=1, keepdims=True)
            
        return M / np.where(l1_norm == 0.0, 1.0, l1_norm)
        
    elif norm_type == "l2":
        if axis == None: 
            l2_norm = np.sqrt(np.sum(M**2))
        elif axis == 0:
            l2_norm = np.sqrt(np.sum(M**2, axis=0, keepdims=True))
        else:
            l2_norm = np.sqrt(np.sum(M**2, axis=1, keepdims=True))

        return M / np.where(l2_norm == 0.0, 1.0, l2_norm)
        
    else:
        if axis == None: 
            max_norm = np.max(np.abs(M))
        elif axis == 0:
            max_norm = np.max(np.abs(M), axis=0, keepdims=True)
        else:
            max_norm = np.max(np.abs(M), axis=1, keepdims=True)

        return M / np.where(max_norm == 0.0, 1.0, max_norm)