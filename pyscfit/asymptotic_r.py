"""Functions to calculate values for the approximate solution to R(t)"""

import scipy.integrate
import scipy.linalg
import numpy as np

def asymptotic_r_vals(q, A, F, tau):
    """Calculate the values used for the asymptotic approximation of R
    
    R is a reliability or survivor function used in the calculation
    of the exact pdf of dwell times with missed events 
    See Hawkes et al (1992) Phil Trans R Soc Lond B pp.393-394
   
    The asymptotic behavior of the matrix function, R, depends on values of
    s that render the matrix W(s) singular. The values of s that render
    W(s) singular are the roots of the determinant of W(s). That is,
          det W(s) = 0
    Jalali and Hawkes (1992) Adv Appl Prob 24 pp.302-321 prove that if the
    Q matrix is irreducible and reversible (which should hold if the gating
    mechanism follows microscopic reversibility - see Colquhoun and Hawkes
    1982 pp. 24-25), then det W(s) = 0 has exactly kA roots
    
    Parameters
    ----------
    q : 2-d array
        The Q matrix
    A : array
        The indices of states in class 1 (open)
    F : array
        The indices of states in class 2 (closed)
    tau : float
        The dead time
    
    Returns
    -------
    s :
    areaR :
    r :
    c :
    Wprime :
    mu :
    a :
    """
    
    TOL = 1e-9
    
    kA = A.shape[0]
    kF = F.shape[0]
    uf = np.ones((kF, 1))
    s = np.full(shape=(kA, 1), fill_value=np.inf)
    r = np.zeros((kA, kA))
    c = np.zeros((kA, kA))
    mu = np.zeros_like(s)
    Wprime = np.zeros((kA, kA, kA))
    areaR = np.zeros((kA, kA, kA))
    a = np.zeros((kA, 1))
    npts = 100*kA + 1
    x0 = np.zeros((1, npts))
    y0 = np.zeros((1, npts))
    idx1 = np.zeros((1, kA))
    idx2 = np.zeros((1, kA))
    
    eqFFt = scipy.linalg.expm(q[np.ix_(F, F)] * tau)
    
    raise NotImplementedError()
    
    return s, areaR, r, c, Wprime, mu, a

def detW(s, q, A, F, tau):
    """Determinant of the matrix function W(s)
    
    See Hawkes, Jalali, & Colquhoun (1990)
    
    Parameters
    ----------
    s : 
    q : 2-d array
        The Q matrix
    A : array
        Indices of states in class 1
    F : array
        Indices of states in class 2
    tau : float
        The dead time (or resolution)
    
    Returns
    -------
    2-d array
        The determinant of W(s)
    """
    
    return scipy.linalg.det(W(s,q,A,F,tau))
    
def W(s, q, A, F, tau):
    """Matrix function W(s)

    Calculates the matrix W from Hawkes et al (1992) Phil Trans R Soc Lond B p.393
    The asymptotic behavior of the matrix function, R, depends on values of
    s that render the matrix W(s) singular.  W(s) is defined as
        $$W(s) = sI - H(s)$$
    where
        $$H(s) = Q_{AA} + Q_{AF} \left ( \int_0^{\tau} e^{-st}e^{Qt}dt \right ) Q_{FA}$$
    or, if s is not an eigenvalue of Q(F,F), then inv(s*I - Q(F,F)) exists
    and
        H(s) = Q(A,A) + Q(A,F)*inv(s*I - Q(F,F))*(I - exp(-(s*I-Q(F,F))*tau))*Q(F,A)    
    
    Parameters
    ----------
    s :
    q :
    A :
    F :
    tau :
    
    Returns
    -------
    W : 2-d array
        Value of W(s)
    """
    
    TOL = 1e-12
    qAA = q[np.ix_(A, A)]
    idA = np.eye(*qAA.shape)
    qAF = q[np.ix_(A,F)]
    qFF = q[np.ix_(F,F)]
    qFA = q[np.ix_(F,A)]
    idF = np.eye(*qFF.shape)
    eig_val_FF, eigFF = scipy.linalg.eig(qFF)

    M = s*idF - qFF
    # check whether (sI-qFF)^-1 exists, which will not occur if sI-qFF is singular, 
    # if sI-qFF is singular, the det(sI-qFF)=0 or equivalently s is an eigenvalue of qFF
    if np.any(np.abs(eigFF-s)<TOL):
        # since (sI-qFF)^-1 does not exist, we cannot shortcut calculation of
        # the integral e^(-(sI-qFF)*t), so let's estimate it numerically
        x = np.linspace(0, tau)
        y = np.zeros(M.shape + (x.size,))
        for i in np.arange(x.size):
            y[:, :, i] = np.real(scipy.linalg.expm(-x[i] * M))
        
        # Numerically estimate the integral using trapz 
        # TODO: assess other integration methods, like scipy.integrate.romb or
        #   scipy.integrate.simps
        #   See https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html#integrating-using-samples
        H1 = scipy.integrate.trapz(y, x, axis=-1)
        H = qAA + qAF @ H1 @ qFA
        # logging.warn('In W(s)\ns is an eigenvalue of qFF, so (sI-qFF)^-1 does not exist.')
    else:
        qAF_M_inv = scipy.linalg.solve(M.T, qAF.T).T
        H = qAA + qAF_M_inv @ (idF - scipy.linalg.expm(-M*tau)) @ qFA

    W = s*idA - H

    return W
    
def dWds( s, q, A, F, tau ):
    """Calculate the derivative of the matrix function W(s)
    
    The derivative of W(s) is used in the asymptotic approximation to the
    matrix function R, which governs the pdf of dwell times when events are
    missed (see Hawkes et al 1990).
 
    This definition of dWds is from Hawkes et al (1992) pp. 394, eq. (56)
    
    Parameters
    ----------
    s : 
    q : 2-d array
        The Q matrix
    A : array
        Indices of states in class 1
    F : array
        Indices of states in class 2
    tau : float
        The dead time
    
    Returns
    -------
    dWds : 2-d array
        Derivative of the matrix function W(s)
    """
    
    qAA = q[np.ix_(A,A)]
    qAF = q[np.ix_(A,F)]
    qFF = q[np.ix_(F,F)]
    qFA = q[np.ix_(F,A)]
    idA = np.eye(qAA.shape)
    idF = np.eye(qFF.shape)
    M = s*idF - qFF
    
    # From eq (2.16) in Hawkes et al (1990), SFF*(s) = I-expm(-(s*I-Q(F,F))*tau)
    SFF = idF - scipy.linalg.expm(-M*tau);
    
    # From eq (4) in Hawkes et al (1992), GFA*(s) = inv(s*I-Q(F,F))*Q(F,A)
    GFA = scipy.linalg.solve(M, qFA)
    
    SFF_M_inv = scipy.linalg.solve(M.T, SFF.T).T

    dWds = idA + qAF @ (SFF_M_inv - tau*(idF-SFF)) @ GFA

    return dWds    