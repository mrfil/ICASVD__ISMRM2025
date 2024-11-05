import numpy as np

def beta_to_tscore(X, Y, beta):
    """
    Converts a beta map to a t-score map for simple contrast of one beta map.
    This assumes that the contrast vector c is a standard basis vector, one c for each beta.
    It also assumes all columns of X are full column rank.

    Parameters:
    X : numpy.ndarray
        Design matrix (n x p)
    Y : numpy.ndarray
        Dependent variable matrix (n x m)
    beta : numpy.ndarray
        Estimated coefficients (p x m)

    Returns:
    t_scores : numpy.ndarray
        T-scores for the betas (p x m)
    """
    n, p = X.shape  # Number of observations, Number of predictors
    m = Y.shape[1]  # Number of dependent variables

    # Compute the residuals
    residuals = Y - X @ beta

    # Estimate the residual variance for each column of Y
    residual_variance = np.sum(residuals**2, axis=0) / (n - p)

    # Compute the variance-covariance matrix for betas
    invXTX = np.linalg.inv(X.T @ X)
    SE = np.sqrt(np.diag(invXTX)[:, np.newaxis] * residual_variance)

    cond_num = np.linalg.cond(X.T @ X)
    if cond_num > 1e15:  # or some other large threshold
        print(f"Warning: X'X is ill-conditioned. Condition number: {cond_num}")
    # Compute t-scores for each column of beta
    
    t_scores = beta / SE

    return t_scores