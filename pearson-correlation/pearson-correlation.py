import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    X = np.asarray(X, dtype=float)
    N = len(X)
    if N < 2 or len(X.shape) != 2:
        return None
        
    Xc = X - np.mean(X, axis=0, keepdims=True)
    
    cov = (Xc.T @ Xc) / (N - 1)
    sigma = np.sqrt(np.diag(cov))
    
    return cov / np.outer(sigma, sigma)

    