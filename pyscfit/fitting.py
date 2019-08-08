"""Functions to fit rates in ion channel gating mechanisms"""

import numpy as np
import scipy.linalg

from qmatrix import cvals, phi, R
from asymptotic_r import asymptotic_r_vals, chs_vectors

def qmatrix_loglik(params, Q, idxtheta, M, b, A, F, tau, dwells):
    """Log likelihood of rates in a gating mechanism
    
    Calculate the log likelihood of rates in an ion channel gating mechanism
    represented by a Q matrix given a sequence of idealized open and closed
    durations.
    
    Parameters
    ----------
    params : array
        The log10-transformed rates to vary
    q : 2-d array
        The Q matrix of transition probabilities
    idxtheta : array
        The indices of `q` that give all the rate constants
    M : 2-d array
        A coefficient matrix for `params` such that M * params + b = \theta
    b : 2-d array
        A constant column vector for the constraints where
        M*params + b = \theta
    A : array
        The initial set of states
    F : array
        The final set of states after the transition
    tau : float
        The dead time or resolution below which no events are detected
    dwells : 1-d array
        Durations of sojourns in each state. The sojourns must alternate
        starting from the set of states A to set F and must end on a dwell
        in set F.
    
    Returns
    -------
    ll : float
        The log likelihood of the rates
    """
    
    if np.any(np.isnan(params)):
        # TODO: issue warning
        return np.nan
    
    if q.ndim < 2 or q.shape[0] != q.shape[1]:
        raise ValueError("q matrix must be a square 2-d matrix")
    
    ndwells = dwells.size / 2
    nstates = q.shape[0]
    qnew = np.zeros(nstates)

    theta = M@params + b
    qnew[idxtheta] = 10.0**theta
    qnew -= diag(qnew.sum(axis=1))

    # Number of multiples of tau to use for exact correction for missed events
    mMax = 2 
    Co, lambdas = cvals(qnew, A, F, tau, mMax)
    so, areaRo = asymptotic_r_vals(qnew, A, F, tau)
    if np.any(np.isinf(so)):
        # TODO: issue warning
        return np.nan
    
    Cs, lambdas_cs = cvals(qnew, F, A, td, mMax)
    s_s, areaRs = asymptotic_r_vals(qnew, F, A, td)
    if np.any(np.isinf(s_s)):
        # TODO: issue warning
        return np.nan
    
    eqAAt = scipy.linalg.expm(qnew[np.ix_(A,A)] * tau)
    eqFFt = scipy.linalg.expm(qnew[np.ix_(F,F)] * tau)
    phiA = phi(qnew, A, F, tau)

    # Calculation of the forward recursions, which are used to calculate the
    # likelihood, will run into numerical problems because the probabilities
    # will almost always be less than one and the likelihood will therefore
    # decay with increasing number of dwell times.  To avoid this, we
    # continually scale the inital probability vector (see Rabiner 1989 A
    # tutorial on hidden Markov models and selected applications in speech
    # recognition Proc. IEEE 77, 257-286 and Qin et al (1997) Maximum
    # likelihood estimation of aggregated Markov processes Proc R Soc Lond B
    # 264, 375-383

    scalefactor = np.zeros((ndwells,1))
    p = phiA
    for ii in np.arange(ndwells):
        t1 = np.abs(dwells[2*ii-1] - tau)
        t2 = np.abs(dwells[2*ii] - tau)
        p = p @ R(Co, lambdas, tau, so, areaRo, mMax, t1) \
              @ qnew[np.ix_(A,F)] @ eqFFt \
              @ R(Cs, lambdas, tau, s_s, areaRs, mMax, t2) \
              @ qnew[np.ix_(F,A)] @ eqAAt
        scalefactor[ii] = 1.0 / sum(p)
        p *= scalefactor[ii]
    
    ll = -(-sum(np.log10(scalefactor)))

    return ll

def qmatrix_loglik_bursts(params, Q, idxtheta, M, b, A, F,
                          tau, tcrit, dwells):
    """Log likelihood of rates in a gating mechanism from bursts of activity
    
    Calculate the log likelihood of rates in an ion channel gating mechanism
    represented by a Q matrix given a series of sequences of idealized
    open and closed durations. Each burst in the series of idealized sequences
    is assumed to be separated by a critical time duration, `tcrit`.
    
    Parameters
    ----------
    params : array
        The log10-transformed rates to vary
    q : 2-d array
        The Q matrix of transition probabilities
    idxtheta : array
        The indices of `q` that give all the rate constants
    M : 2-d array
        A coefficient matrix for `params` such that M * params + b = \theta
    b : 2-d array
        A constant column vector for the constraints where
        M*params + b = \theta
    A : array
        The initial set of states
    F : array
        The final set of states after the transition
    tau : float
        The dead time or resolution below which no events are detected
    tcrt : float
        The critical time that separates each burst in the series.
    dwells : list of 1-d arrays
        Durations of sojourns in each state. The sojourns must alternate
        starting from the set of states A to set F and must end on a dwell
        in set F. Each element of the list corresponds to one burst of ion
        channel activity, and the elements of the list are assumed to be
        separated by at least `tcrit`.
    
    Returns
    -------
    ll : float
        The log likelihood of the rates
    """
    
    if np.any(np.isnan(params)):
        # TODO: issue warning
        return np.nan
    
    if q.ndim < 2 or q.shape[0] != q.shape[1]:
        raise ValueError("q matrix must be a square 2-d matrix")
    
    n_bursts = len(dwells)
    n_states = q.shape[0]
    qnew = np.zeros_like(q)
    
    # Number of multiples of tau to use for exact correction for missed events
    mMax = 2
    
    Co, lambdas = cvals(qnew, A, F, tau, mMax)
    so, areaRo = asymptotic_r_vals(qnew, A, F, tau)
    if np.any(np.isinf(so)):
        # TODO: issue warning
        return np.nan
    
    Cs, lambdas_cs = cvals(qnew, F, A, td, mMax)
    s_s, areaRs = asymptotic_r_vals(qnew, F, A, td)
    if np.any(np.isinf(s_s)):
        # TODO: issue warning
        return np.nan
    
    eqAAt = scipy.linalg.expm(qnew[np.ix_(A,A)] * tau)
    eqFFt = scipy.linalg.expm(qnew[np.ix_(F,F)] * tau)

    # Calculation of the forward recursions, which are used to calculate the
    # likelihood, will run into numerical problems because the probabilities
    # will almost always be less than one and the likelihood will therefore
    # decay with increasing number of dwell times.  To avoid this, we
    # continually scale the inital probability vector (see Rabiner 1989
    # and Qin et al (1997) Proc R Soc Lond B

    phib, ef = chs_vectors(qnew, A, F, areaRs, -1/s_s, tau, tcrit)
    LL = np.zeros(n_bursts)
    for ii in np.arange(n_bursts):
        tmpdwells = dwells[ii]
        numdwells = tmpdwells.size
        pA = phib
        pF = np.zeros((1, F.size))
        scalefactor = np.zeros(numdwells)
        for jj in np.arange(numdwells):
            dwell_time = tmpdwells[jj] - tau
            if j % 2 == 0:
                pF = pA @ R(Co,lambdas,tau,so,areaRo,mMax,dwell_time) \
                        @ qnew[np.ix_(A,F)] \
                        @ eqFFt
                scalefactor[jj] = 1./sum(pF)
                pF *= scalefactor[jj]
            else
                pA = pF @ R(Cs,lambdas,tau,s_s,areaRs,mMax,dwell_time) \
                        @ qnew[np.ix_(F,A)] \
                        @ eqAAt
                scalefactor[jj] = 1./sum(pA)
                pA *= scalefactor[jj]
        LL[ii] = -sum(np.log10(scalefactor)) + log10(pF @ ef)
    ll = -sum(LL)

    return ll
