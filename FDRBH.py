import numpy as np

def FDRBH(P, alpha):
    """
    Algorithm for the False Discovery Rate (FDR) for false positive correction
    Using the Benjamini-Hochberg procedure

    Warning: This uses a BH method. May need verification for replication.

    Parameters:
    P : numpy.ndarray
        The p-values of all the hypotheses
    alpha : float
        Upper bound of the false discovery rate (FDR)

    Returns:
    positives : numpy.ndarray
        Boolean array indicating which hypotheses are considered significant

    Reference:
    https://en.wikipedia.org/wiki/False_discovery_rate
    """
    P_sort_tmp = np.sort(P)
    
    # Don't count pixels that were 'removed' from data (brain extraction for example)
    m = len(P) - np.sum(np.isnan(P))
    P_sort = P_sort_tmp[:m]
    
    k = np.arange(1, m + 1)
    thresh = k * alpha / m
    A = P_sort <= thresh
    
    # Get largest index of A that == True (largest k)
    largest_k = np.where(A)[0][-1] if np.any(A) else 0
    
    # When nothing is declared as significant, positives is defined to be all False
    if largest_k == 0:
        positives = np.zeros_like(P, dtype=bool)
    else:
        positives = P <= P_sort[largest_k]
    
    return positives