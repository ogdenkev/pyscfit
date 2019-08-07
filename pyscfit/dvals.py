import scipy.linalg
import numpy as np

def dvals(q, A, F, td):
    """Calculate intermediate values for reliability function
    
    The probability that an e-open time starting in state i has not
    finished and is currently in state j is given by equation 3.1 in
    Hawkes, Jalali, and Colquhoun (1990). This function, called R(t),
    is a matrix function. Equation 3.11 of HJC (1990) gives R(t) as an
    infinite series of m-fold convolutions.
    
    HJC prove that the m-fold convolutions can be expressed in terms of
    a product of a polynomial of degree m in t (with matrix-valued
    coefficients) and an exponential. The polynomial coefficients depend
    on the spectral matrices of Q and the sub-matrix exponential.
    
    This function returns the product of the spectral matrices of Q and
    the sub-matrix exponential. It is equation 3.16 in Hawkes, Jalali, and
    Colquhoun (1990) Phil. Trans. R. Soc. Lond. A.
    
        $$ D_j = A_{jAF}e^{Q_{FF}\tau}Q_{FA} $$
    
    where A_j are the spectral matrices of Q
    
    Parameters
    ----------
    q : matrix
        The Q matrix
    A : array
        Indices of states in Q that are open states
    F : array
        Indices of shut states in Q
    td : float
        Dead time
    
    Returns
    -------
    D : 
    taus :
    lambda : 
    specA :
    """
    
    # The indices need to be reshaped to work as in Matlab
    # See the second example at https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#purely-integer-array-indexing
    # Either q[A[:, np.newaxis], A] or q[np.ix_(A, A)]
    # or q[A[:, None], A]
    qAA = q[np.ix_(A,A)]
    qFF = q[np.ix_(F,F)]
    qAF = q[np.ix_(A,F)]
    qFA = q[np.ix_(F,A)]
    taus, lambdas, specA = qmatvals(q)
    eQFFt = scipy.linalg.expm(qFF*td)
    nA = A.shape[0]
    nQ = q.shape[0]
    D = np.zeros((nA, nA, nQ))

    for ii in np.arange(nQ):
        D[:, :, ii] = specA[A, F, ii] @ eQFFt @ qFA
    
    return D, taus, lambdas, specA