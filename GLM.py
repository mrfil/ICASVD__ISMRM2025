import numpy as np

def GLM(X, Y):
    """
    Solves the General Linear Model (GLM)
    
    Parameters:
    X : numpy.ndarray
        Design matrix (n x p)
    Y : numpy.ndarray
        Dependent variable (n x 1)
    
    Returns:
    beta : numpy.ndarray
        Estimated parameters (p x 1)
    
    Raises:
    ValueError: If the number of rows in X and Y are not the same
    """
    # Check for inputs
    if X.shape[0] != Y.shape[0]:
        raise ValueError('The number of rows in X and Y must be the same.')
    
    # Compute the beta vector using the least squares solution
    beta = np.linalg.solve(X.T @ X, X.T @ Y)
    
    return beta




