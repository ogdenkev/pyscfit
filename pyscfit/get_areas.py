def get_areas(A, p0):
    """Calculate the exponential mixture areas from spectral matrices
    
    Parameters
    ----------
    A : array
        Spectral matrices of the Q matrix. Should be 3-dimensional
    p0 : 1-d array
        Initial probabilities of each state.
    
    Returns
    -------
    areas : 2-d array
        The areas for the components of the exponential mixture
        distrubtion corresponding to the Q matrix
    """
    if p0.ndim == 1 or p0.shape[1] == 1:
        p0 = p0.T
    
    n_states = p0.shape[0]
    areas = np.zeros((n_states, n_states))
    for r in np.arange(n_states):
        for bb in np.arange(1, n_states):
            areas[r, bb] = p0 @ A[:, r, bb]
    
    areas[:, 1] = []
    
    return areas
