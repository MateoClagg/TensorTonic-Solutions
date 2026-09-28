import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    """
    Returns the inverse as a NumPy array, or None.
    """
    A = np.asarray(A, dtype=float)
    n = len(A)
    aug_M = np.concatenate((A, np.eye(n)), axis=1)

    for col in range(n):
        # search values below current for bigger absolute pivot
        max_pivot = col + np.abs(aug_M[col:, col]).argmax() 
        if aug_M[max_pivot, col] < 1e-12: # return None if no nonzero pivot exists
            return None 
        
        if col != max_pivot:
            aug_M[[col, max_pivot]] = aug_M[[max_pivot, col]] # swap rows to get max pivot
        pivot = aug_M[col, col]

        # scale pivot row to 1
        aug_M[col] = aug_M[col] / aug_M[col, col]

        # eliminate values above and below pivot
        for row in range(n):
            if row != col:
                aug_M[row] = aug_M[row] - aug_M[row, col] * aug_M[col]

    return aug_M[:, n:]

        
            
        
        
        
        
        