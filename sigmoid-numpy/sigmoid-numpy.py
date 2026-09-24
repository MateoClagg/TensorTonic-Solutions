import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    if isinstance(x, (float, int)):
        return 1 / (1 + np.exp(-x))
    elif isinstance(x, list):
        x = np.array(x)
        return 1 / (1 + np.exp(-x))
    else:
        return None
        
    
        