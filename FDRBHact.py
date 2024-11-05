

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
from GLM import GLM

# Assuming GLM, FDRBH, and beta_to_tscore functions have been defined earlier

def FDRBHact(img, FWHM, q, X):
    """
    This function is to pass perfusion fMRI data through to see which voxels
    are active for task-based activation.

    Parameters:
    img : numpy.ndarray
        The 4D time series, 4th dimension is time.
    FWHM : float
        FWHM of the Gaussian for smoothing data
    q : float
        Upper bound for expected value of false discovery rate for FDRBH algorithm
    X : numpy.ndarray
        Design matrix for the GLM

    Returns:
    log10p : numpy.ndarray
        The -log10 of the 1-sided p values for correlations with task. A
        map of the positives only with the 1st 3 dimensions the brain volume and
        the 4th dimension the number of correlates from the design matrix.
    """
    from GLM import GLM
    from beta_to_tscore import beta_to_tscore
    from FDRBH import FDRBH

    img = np.maximum(img, 0)
    ny, nx, nz, nt = img.shape

    if FWHM != 0:  # add spatial smoothing
        std = FWHM / (2.4 * 3)  # mm*pixels/mm=pixels
        for i in range(img.shape[3]):
            img[:,:,:,i] = gaussian_filter(img[:,:,:,i], sigma=std)

    n = img.shape[3]

    # Run GLM
    Y = img.reshape(nx*ny*nz, nt).T
    beta = np.zeros((X.shape[1], nx*ny*nz))

    for i in range(nx*ny*nz):
        beta[:, i] = GLM(X, Y[:, i])

    t = beta_to_tscore(X, Y, beta)  # get t-scores

    # Look at data
    t_map = np.zeros((nx, ny, nz, X.shape[1]))
    for i in range(X.shape[1]):
        t_map[:,:,:,i] = t[i,:].reshape(nx, ny, nz)

    pval1side = 1 - stats.t.cdf(t_map, n-2)

    # Use an FDR method
    log10p = np.zeros_like(pval1side)
    for i in range(X.shape[1]):
        p = pval1side[:,:,:,i]
        positives = FDRBH(p.flatten(), q)
        activation = positives.reshape(ny, nx, nz)
        #log10p[:,:,:,i] = -np.log10(pval1side[:,:,:,i]) * activation
        log10p[:,:,:,i] = -np.log10(np.maximum(pval1side[:,:,:,i], np.finfo(float).tiny)) * activation
    
    return log10p

    #return log10p



