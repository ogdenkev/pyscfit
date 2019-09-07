"""Q Matrix Functions"""

import scipy.linalg
import numpy as np


def qmatvals(q):
    """Calculate the time constants and spectral matrices for a Q matrix
    
    The spectral matrices of Q are used to calculate the areas of the
    componenets of the exponential mixture distribution.
    
    Parameters
    ----------
    q : matrix
        Q matrix, a k-by-k matrix where k is the number of states and 
        q[i, j] is the transition rate from state i to state j and
        q[i, i] is a number such that the sum of each row is zero
    
    Returns
    -------
    taus : array
        time constant for macroscopic fluctuations
    lambda : array
        eigenvalues of q; lambda[i] corresponds to A[:, :, i]
    A : matrix
        spectral matrices of q; 3-dimensional
    """

    TOL = 1e-12

    lambdas, v = scipy.linalg.eig(-q)
    if not np.all(lambdas.imag == 0):
        raise ValueError("Eigenvalues of Q are not real")
    lambdas = lambdas.real
    v = v.real
    y = scipy.linalg.inv(v)
    A = np.zeros(v.shape + (q.shape[0],))
    for k in np.arange(q.shape[0]):
        A[:, :, k] = v[:, k, np.newaxis] @ y[np.newaxis, k, :]

    zero_mask = np.abs(lambdas) < TOL
    lambdas[zero_mask] = 0
    idx = np.argsort(lambdas, axis=0)
    lambdas = np.take_along_axis(lambdas, idx, axis=0)
    taus = np.zeros_like(lambdas)
    zero_mask = np.abs(lambdas) < TOL
    taus[~zero_mask] = 1.0 / lambdas[~zero_mask]
    taus[zero_mask] = np.inf
    A = A[:, :, idx]

    # if an eigenvalue of Q is zero, then there is not a corresponding tau
    # taus[lambda==0] = 0

    return taus, lambdas, A


def dvals(q, A, F, td, spec_mat):
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
    spec_mat : array
        Spectral matrices of q. Must be 3-dimensional
    
    Returns
    -------
    D : 3-d array
        The intermediate values needed for polynomial coefs
    """

    # The indices need to be reshaped to work as in Matlab
    # See the second example at https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#purely-integer-array-indexing
    # Either q[A[:, np.newaxis], A] or q[np.ix_(A, A)]
    # or q[A[:, None], A]
    qAA = q[np.ix_(A, A)]
    qFF = q[np.ix_(F, F)]
    qAF = q[np.ix_(A, F)]
    qFA = q[np.ix_(F, A)]

    eQFFt = scipy.linalg.expm(qFF * td)

    nA = A.shape[0]
    nQ = q.shape[0]
    D = np.zeros((nA, nA, nQ))

    # Indexing spec_mat with np.ix_(A, F, [ii]) creates a 3-dim array
    # so it would need to be squeezed. So we'll use the np.newaxis style
    for ii in np.arange(nQ):
        D[:, :, ii] = spec_mat[A[:, None], F, ii] @ eQFFt @ qFA

    return D


def cvals(q, A, F, td, mMax=2):
    """Calculate matrix-valued coefficients used in reliability function R
    
    This function calculates the matrix-valued coefficients called C in
    the Theorem on page 519 of Hawkes, et al. (1990) Phil. Trans. R. Soc.
    Lond. A. These coefficients are used for the polynomial of degree m in
    t, which Hawkes et al. refer to as B.
    
    Parameters
    ----------
    q : matrix
        The Q-matrix
    A : array
        Indices of states in Q matrix that are open states
    F : array
        Indices of states in Q matrix that are shut states
    td : float
        Dead time or resolution
    mMax : int
        Number of dead times to use exact correction for missed events
    
    Returns
    -------
    C : array
        The matrix-valued coefficients for R(t)
    """

    nstates = q.shape[0]
    nA = A.shape[0]
    taus, lambdas, specA = qmatvals(q)
    D = dvals(q, A, F, td, specA)
    C = np.zeros((nA, nA, nstates, mMax + 1, mMax + 1))
    sAaa = specA[A[:, None], A, :]

    for i in np.arange(nstates):
        for m in np.arange(mMax + 1):
            if m == 0:
                C[:, :, i, m, m] = sAaa[:, :, i]
            else:
                C[:, :, i, m, m] = D[:, :, i] @ C[:, :, i, m - 1, m - 1] / m

    for m in np.arange(mMax + 1):
        for n in np.arange(m):
            for i in np.arange(nstates):
                tmp = np.zeros((nA, nA))
                if n == 0:
                    for j in np.arange(nstates):
                        if i == j:
                            continue
                        for r in np.arange(m):
                            tmp += (
                                D[:, :, i]
                                @ C[:, :, j, m - 1, r]
                                * np.math.factorial(r)
                                / (lambdas[j] - lambdas[i]) ** (r + 1)
                            )
                            tmp -= (
                                D[:, :, j]
                                @ C[:, :, i, m - 1, r]
                                * np.math.factorial(r)
                                / (lambdas[i] - lambdas[j]) ** (r + 1)
                            )
                    C[:, :, i, m, n] = tmp
                else:
                    for j in np.arange(nstates):
                        if i == j:
                            continue
                        for r in np.arange(n, m):
                            tmp += (
                                D[:, :, j]
                                @ C[:, :, i, m - 1, r]
                                * np.math.factorial(r)
                                / (
                                    n
                                    * (lambdas[i] - lambdas[j]) ** (r - n + 1)
                                )
                            )
                    C[:, :, i, m, n] = (
                        D[:, :, i] @ C[:, :, i, m - 1, n - 1] / n - tmp
                    )

    return C


def R(t, C, lambdas, tau, s, areaR, mMax=2):
    """Calculate the value of the matrix function R(t)
    
    R(t) is a kind of reliability or survivor function, in which R(i,j)
    gives the probability that a resolved open time, starting in state i,
    has not yet finished and is currently in state j.  Another way to state
    this is that R(i,j)[t] = the probability that 1)you're in state j and 
    2)there has been no resolvable shut time during the interval 0 to t
    given that you were in state i at time 0.
 
    For details see Hawkes et al (1990, 1992)
    
    Parameters
    ----------
    C : array
        C is used to calculate the exact value of R for times less than or
        equal to twice the imposed resolution
    lambdas : array
        Eigenvalues of the Q matrix
    tau : float
        The imposed resolution
    s : array
        Generalized eigenvalues used for asymptotic approximation to R(t)
        The s values can be calculated from the function asymptoticRvals 
    areaR : array
        The area of each exponential component given in s
    t : float
        The time at which to return R(t)
    
    Returns
    -------
    R : 2-d array
        The reliability/survivor function R(t) evaluated at time t
    """

    TOL = 1e-12

    nr, nc = areaR.shape
    f = np.zeros((nr, nc))

    if t < 0:
        return f

    if t <= mMax * tau:
        # Exact correction for missed events
        kA = lambdas.size
        m = np.ceil(t / tau) - 1
        if np.abs(t) < TOL:
            m = 0

        for i in np.arange(m + 1):
            for j in np.arange(kA):
                for k in np.arange(i):
                    # Beware of the indexing!!!! KKO 140923
                    f += np.real(
                        (-1) ** i
                        * C[:, :, j, i, k]
                        * (t - ii * tau) ** kk
                        * np.exp(-lambdas[jj] * (t - ii * tau))
                    )
    else:
        # Approximate correction for missed events
        kA = s.size
        for ii in np.arange(kA):
            f += np.real(np.exp(s[ii] * t) * areaR[:, :, ii])

    return f


def eG(q, A, F, tau, s):
    """Laplace transform of the corrected probability density matrix
    
    A semi-Markov process occurs when the Markov process with the
    transition rate matrix Q switches from the set of states A to the
    set of states F. This function gives the Laplace transform of the
    matrix of probability densities of the intervals in the case when all
    events of duration less than tau are missed.
    
    See equations 2.12, 2.19, and 2.20 of HJC1990.
    
    Parameters
    ----------
    q : 2-d array
        The Q matrix of transition probabilities
    A : array
        The initial set of states
    F : array
        The final set of states after the transition
    tau : float
        The dead time or resolution, below which all events are missed
    s : complex
        The Laplace variable, s, which is a complex number
    
    Returns
    -------
    eG : 2-d array
        The Laplace transform of the probability density matrix when events
        shorter than tau duration are missed.
    """

    nA = A.size
    nF = F.size
    I_AA = np.eye(nA)
    I_FF = np.eye(nF)

    qAA = q[np.ix_(A, A)]
    qAF = q[np.ix_(A, F)]
    qFF = q[np.ix_(F, F)]
    qFA = q[np.ix_(F, A)]

    # See equations 2.9 and 2.10 of HJC1990
    Gstar_AF = scipy.linalg.solve(s * I_AA - qAA, qAF)
    Gstar_FA = scipy.linalg.solve(s * I_FF - qFF, qFA)

    prob_tau_to_inf = scipy.linalg.expm(-(s * I_FF - qFF) * tau)
    # See equations 2.16 and 2.18 in HJC1990
    short_FF = I_FF - prob_tau_to_inf
    # See equations 2.17 and 2.18 in HJC1990
    long_FF = prob_tau_to_inf

    numer = I_AA - Gstar_AF @ short_FF @ Gstar_FA
    denom = Gstar_AF @ long_FF
    y = scipy.linalg.solve(numer, denom)

    return y


def phi(q, A, F, tau):
    """Equilibrium vector for the semi-Markov process
    
    A semi-Markov process is embedded in the continuous time Markov chain
    with transition rate matrix Q when that process changes from the set
    of states A to the set of states F. This function gives the equilibrium
    vector of that semi-Markov process.
    
    See equations 2.14 of HJC1990.
    Also see Hawkes and Sykes (1990) IEEE Trans on Reliability
    
    Parameters
    ----------
    q : 2-d array
        The Q matrix of transition probabilities
    A : array
        The initial set of states
    F : array
        The final set of states after the transition
    tau : float
        The dead time or resolution below which no events are detected
    
    Returns
    -------
    phi : 1-d array
        The equilibrium state vector
    """

    # phi is the solution to the equation
    # phi @ C = zF, or equivalently, C.T @ phi.T = zF.T
    # and C is an augmented matrix
    # where zF is the augmented matrix [0, 0, ... , 0_nstates, 1]

    nA = A.size
    uF = np.ones((nA, 1))
    C = np.eye(nA) - eG(q, A, F, tau, 0) @ eG(q, F, A, tau, 0)
    C = np.concatenate((C, uF), axis=1)
    zF = np.concatenate((np.zeros((1, nA)), np.ones((1, 1))), axis=1)
    phi, residues, rank, singular_vals = scipy.linalg.lstsq(C.T, zF.T)
    return phi.T
