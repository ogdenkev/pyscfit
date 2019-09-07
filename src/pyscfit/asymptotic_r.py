"""Functions to calculate values for the approximate solution to R(t)"""

# Need scipy>=1.2.0 for optimize.root_scalar
import scipy.integrate
import scipy.linalg
import scipy.optimize
import numpy as np

from qmatrix import phi


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
    # We want to interpolate each interval [detW[i-1], detW[i]] with 100 points where
    # detW[-1] = 0 (in other words, prepend detW with 0) and detW.size should be kA
    # Then let delta = k * ((detW[i] - detW[i-1]) / 100) where k = 0, 1, ..., 100
    # which is a total of 101 points
    # Thus there are kA+1 actual points comprising kA regions to interpolate,
    # and each region requires 101 points, but the 100-th point of region v is
    # the same as the 0-th point of region v-1 so the total points needed is
    # num_regions * (points_per_region+1) - (num_points - 2)
    # = kA*(100+1) - ((kA+1) - 2) = 100kA + kA - kA - 1 + 2 = 100kA + 1
    pts_per_region = 100
    npts = pts_per_region * kA + 1
    x0 = np.zeros(npts)
    y0 = np.zeros(npts)
    idx1 = np.zeros(kA, dtype="int")
    idx2 = np.zeros(kA, dtype="int")

    eqFFt = scipy.linalg.expm(q[np.ix_(F, F)] * tau)

    # Find the kA roots of det W(s) = 0
    # First, guess that the roots are the same as the tau^-1 for the uncorrected
    # distribution
    D, V = scipy.linalg.eig(q[np.ix_(A, A)])
    # Add a zero to the start of the eigenvalues and then convert the array
    # to a column vector from a 0-d array
    D = np.concatenate((np.zeros(1), D)).reshape(-1, 1)
    tempx = np.real(np.sort(D, axis=0))

    #  Start with the values of tempx closest to zero, which correspond to time
    #  constants with the longest duration since tau ~= 1 ./ s
    #  This will allow us to (hopefully) find the time constants that are above
    #  the imposed resolution
    #  KKO 27 Aug 2014
    tempx = tempx[::-1, :]  # flipud(tempx)

    #  What if all eigenvalues are the same??? Does det W still have kA roots?
    #  Is the Q matrix then reducible?

    #  Construct initial guesses by evenly spacing 100 points in between the
    #  values in tempx
    #  Then find when the sign of detW(x0) changes and save that index for later
    #  input to rootsdetW
    uu = 0
    for i_region in np.arange(kA):
        left = tempx[i_region]
        right = tempx[i_region + 1]
        dx = (right - left) / pts_per_region
        for i_pt in np.arange(pts_per_region + 1):
            if i_pt == 0 and i_region > 0:
                continue
            ind = i_region * (pts_per_region) + i_pt
            x0[ind] = dx * i_pt + left
            y0[ind] = np.real(detW(x0[ind], q, A, F, tau))
            if ind > 0:
                if y0[ind] > 0 and y0[ind - 1] < 0:
                    idx1[uu] = ind - 1
                    idx2[uu] = ind
                    uu += 1
                elif y0[ind] < 0 and y0[ind - 1] > 0:
                    idx1[uu] = ind - 1
                    idx2[uu] = ind
                    uu += 1
            if uu >= kA:
                break
        if uu >= kA:
            break

    ii = 0
    x0 = np.real(x0)
    for rr in np.arange(uu):
        bracket = [x0[idx1[rr]], x0[idx2[rr]]]
        sol = scipy.optimize.root_scalar(
            detW, bracket=bracket, args=(q, A, F, tau)
        )
        if np.all(np.abs(sol.root - s) > TOL):
            s[ii] = sol.root
            ii += 1

    # If not all roots have been found, try using fzeroKO (in rootsdetW) using
    # the first and last values of x0 as inputs
    if np.any(np.isinf(s)):
        ii = np.nonzero(np.isinf(s))[0]
        sol = scipy.optimize.root_scalar(detW, x0=x0[-1], args=(q, A, F, tau))
        if np.all(np.abs(sol.root - s) > TOL):
            s[ii] = sol.root

    if np.any(np.isinf(s)):
        ii = np.nonzero(np.isinf(s))[0]
        sol = scipy.optimize.root_scalar(detW, x0=x0[1], args=(q, A, F, tau))
        if np.all(np.abs(sol.root - s) > TOL):
            s[ii] = sol.root

    # If we still haven't found all roots, then give up
    if np.any(np.isinf(s)):
        return

    # Calculate the right and left eigenvectors of H(s), where s are the roots
    # of det W(s) = 0, which are also eigenvalues of H(s)
    #
    # The left eigenvector, r, is a solution to rW(s) = 0, ru=1, where u is a
    # vector of ones - this is similar to finding equilibrium vector of a Q
    # matrix (e.g. Hawkes and Sykes 1990)
    #
    # The right eigenvector, c, is a solution to c'W(s)' = 0, c'u =1

    for ii in np.arange(kA):
        _Ws = W(s[ii], q, A, F, tau)
        numer = np.concatenate((_Ws, np.ones((kA, 1))), axis=1)
        denom = np.concatenate((np.zeros((1, kA)), np.ones((1, 1))), axis=1)
        r_sol, res, rank, singular_vals = scipy.linalg.lstsq(numer.T, denom.T)
        r[ii, :] = r_sol.T
        numer = np.concatenate((_Ws.T, np.ones((kA, 1))), axis=1)
        c_sol, res, rank, singular_vals = scipy.linalg.lstsq(numer.T, denom.T)
        c[ii, :] = c_sol.T
    c = c.T

    mu = -1.0 / s
    for ii in np.arange(kA):
        Wprime[:, :, ii] = dWds(s[ii], q, A, F, tau)
        areaR[:, :, ii] = (
            c[:, ii, None]
            @ r[None, ii, :]
            / (r[None, ii, :] @ Wprime[:, :, ii] @ c[:, ii, None])
        )
        a[ii] = (
            mu[ii]
            * phi(q, A, F, tau)
            @ c[:, ii, None]
            @ r[None, ii, :]
            @ q[np.ix_(A, F)]
            @ eqFFt
            @ uf
            / (r[None, ii, :] @ Wprime[:, :, ii] @ c[:, ii, None])
        )

    return s, areaR, r, c, Wprime, mu, a


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
    qAF = q[np.ix_(A, F)]
    qFF = q[np.ix_(F, F)]
    qFA = q[np.ix_(F, A)]
    idF = np.eye(*qFF.shape)
    eig_val_FF, eigFF = scipy.linalg.eig(qFF)

    M = s * idF - qFF
    # check whether (sI-qFF)^-1 exists, which will not occur if sI-qFF is singular,
    # if sI-qFF is singular, the det(sI-qFF)=0 or equivalently s is an eigenvalue of qFF
    if np.any(np.abs(eigFF - s) < TOL):
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
        H = qAA + qAF_M_inv @ (idF - scipy.linalg.expm(-M * tau)) @ qFA

    W = s * idA - H

    return W


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

    return scipy.linalg.det(W(s, q, A, F, tau))


# def rootsdetW(q, A, F, tau, x0):
#     """Roots of the determinant of W(s)"""
#     y = scipy.optimize.newton(detW, x0, args=(q, A, F, tau))


def dWds(s, q, A, F, tau):
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

    qAA = q[np.ix_(A, A)]
    qAF = q[np.ix_(A, F)]
    qFF = q[np.ix_(F, F)]
    qFA = q[np.ix_(F, A)]
    idA = np.eye(*qAA.shape)
    idF = np.eye(*qFF.shape)
    M = s * idF - qFF

    # From eq (2.16) in Hawkes et al (1990), SFF*(s) = I-expm(-(s*I-Q(F,F))*tau)
    SFF = idF - scipy.linalg.expm(-M * tau)

    # From eq (4) in Hawkes et al (1992), GFA*(s) = inv(s*I-Q(F,F))*Q(F,A)
    GFA = scipy.linalg.solve(M, qFA)

    SFF_M_inv = scipy.linalg.solve(M.T, SFF.T).T

    dWds = idA + qAF @ (SFF_M_inv - tau * (idF - SFF)) @ GFA

    return dWds


def chs_vectors(q, A, F, areaR, mu, tau, tcrit):
    """Gives the initial and final vectors used for rate MLE from bursts
    
    The initial vector, phib, and the final vector, ef, are used in the
    maximum likelihood estimation of single channel rate constants when the
    input data are bursts of channel activity. These vectors are described
    by Colquhoun, Hawkes, and Srodzinski (1996) Phil Trans R Soc Lond A,
    equations 5.8 and 5.11
    
    Parameters
    ----------
    q : 2-d array
        The Q matrix of transition probabilities
    A : array
        The initial set of states
    F : array
        The final set of states after the transition
    areaR : 3-d array
        The matrix used to determine the asymptotic approximation of the
        distribution of e-opens and e-shuts. If A corresponds to the open
        states, then areaR should correspond to the shut states and vice
        versa. areaR can be calculated from `asymptotic_r_vals`
    mu : 1-d array
        Time constants for the asymptotic shut time distribution - this is
        returned from `asymptoticRvals`.
    tau : float
        Resolution (or dead time) imposed on the data
    tcrit : float
        Critical gap length separating bursts of activity arising from a
        single ion channel
    
    Returns
    -------
    phib : 1-d array
        The initial vectors for MLE of the rates
    ef : 1-d array
        The final vector to use for MLE of the rates
    """

    kA = A.size
    kF = F.size
    uA = np.ones((kA, 1))
    Hfa = np.zeros((kF, kA))

    # const.shape == (kF, kA)
    const = q[np.ix_(F, A)] @ scipy.linalg.expm(q[np.ix_(A, A)] * tau)
    time = tcrit - tau

    phiF = phi(q, A, F, tau)

    if areaR.shape != (kF, kF, kF):
        raise ValueError(
            "areaR.shape, {}, must be (kF, kF, kF) where kF = F.size, but"
            "F.size is {}".format(areaR.shape, kF)
        )

    for i in np.arange(kF):
        Hfa += areaR[:, :, i] @ const * mu[i] * np.exp(-time / mu[i])

    ef = Hfa @ uA
    numer = phiF @ Hfa @ uA
    denom = phiF @ Hfa
    phib = scipy.linalg.solve(numer.T, denom.T).T

    return phib, ef
