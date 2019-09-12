"""Functions to fit rates in ion channel gating mechanisms"""

import warnings
import numpy as np
import scipy.linalg
import scipy.sparse.csgraph

from .qmatrix import cvals, phi, R
from .asymptotic_r import asymptotic_r_vals, chs_vectors
from .utils import _match_hash


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

    theta = M @ params + b
    qnew[idxtheta] = 10.0 ** theta
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

    eqAAt = scipy.linalg.expm(qnew[np.ix_(A, A)] * tau)
    eqFFt = scipy.linalg.expm(qnew[np.ix_(F, F)] * tau)
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

    scalefactor = np.zeros((ndwells, 1))
    p = phiA
    for ii in np.arange(ndwells):
        t1 = np.abs(dwells[2 * ii - 1] - tau)
        t2 = np.abs(dwells[2 * ii] - tau)
        p = (
            p
            @ R(Co, lambdas, tau, so, areaRo, mMax, t1)
            @ qnew[np.ix_(A, F)]
            @ eqFFt
            @ R(Cs, lambdas, tau, s_s, areaRs, mMax, t2)
            @ qnew[np.ix_(F, A)]
            @ eqAAt
        )
        scalefactor[ii] = 1.0 / sum(p)
        p *= scalefactor[ii]

    ll = -(-sum(np.log10(scalefactor)))

    return ll


def qmatrix_loglik_bursts(params, Q, idxtheta, M, b, A, F, tau, tcrit, dwells):
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

    theta = M @ params + b
    qnew[idxtheta] = 10.0 ** theta
    qnew -= np.diag(qnew.sum(axis=1))

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

    eqAAt = scipy.linalg.expm(qnew[np.ix_(A, A)] * tau)
    eqFFt = scipy.linalg.expm(qnew[np.ix_(F, F)] * tau)

    # Calculation of the forward recursions, which are used to calculate the
    # likelihood, will run into numerical problems because the probabilities
    # will almost always be less than one and the likelihood will therefore
    # decay with increasing number of dwell times.  To avoid this, we
    # continually scale the inital probability vector (see Rabiner 1989
    # and Qin et al (1997) Proc R Soc Lond B

    phib, ef = chs_vectors(qnew, A, F, areaRs, -1 / s_s, tau, tcrit)
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
                pF = (
                    pA
                    @ R(Co, lambdas, tau, so, areaRo, mMax, dwell_time)
                    @ qnew[np.ix_(A, F)]
                    @ eqFFt
                )
                scalefactor[jj] = 1.0 / sum(pF)
                pF *= scalefactor[jj]
            else:
                pA = (
                    pF
                    @ R(Cs, lambdas, tau, s_s, areaRs, mMax, dwell_time)
                    @ qnew[np.ix_(F, A)]
                    @ eqAAt
                )
                scalefactor[jj] = 1.0 / sum(pA)
                pA *= scalefactor[jj]
        LL[ii] = -sum(np.log10(scalefactor)) + log10(pF @ ef)
    ll = -sum(LL)

    return ll


def constrain_qr(gamma, idxall, idxvary):
    """QR factorization to separate free from constrained rates

    Given a set of constraints, gamma, on the log10-transformed rates in a
    gating mechanism, theta, such that gamma @ theta == xi, where xi is a
    constant vector, this function will partition gamma into the
    contrained rates, R1, and the free rates, R2.

    If the linear constraints on all the parameters mu, where
    mu(i,j) = log10(q(i,j)), are represented in the matrix gamma and
                              gamma * theta = xi
    where theta = (...mu(i,j)...)' and xi is a constant vector
    Then theta can be formulated into linear combinations of
    unconstrained variables using the QR factorization of gamma

    It is assumed the constraints are independent of one another, which means
    gamma has full rank

    See Qin et al. 1996 Biophys J or
    Golub and Van Loan 1989 Matrix Computations

    Parameters
    ----------
    gamma : 2-d array
        pass
    idxall : array
        pass
    idxvary : array
        pass

    Returns
    -------
    U : 2-d array
    R1 : 2-d array
    R2 : 2-d array
    idxtheta : array
    Gamma : 2-d array
    R : 2-d array
    idxconstrain : array
    n_constrain : float
    """

    if not isinstance(idxall, np.ndarray):
        raise ValueError("idxall must be a numpy array")
    if not isinstance(idxvary, np.ndarray):
        raise ValueError("idxvary must be a numpy array")

    if idxall.ndim != 1:
        raise ValueError("idxall must be 1-d")
    if idxvary.ndim != 1:
        raise ValueError("idxvary must be 1-d")

    n_rates = idxall.size
    n_vary = idxvary.size
    n_constrain = n_rates - n_vary

    # Need to generate theta such that theta = log10([q(idxconstrain); q(idxvary)]);
    # Generating theta doesn't actually need to be done here, but we need to
    # arrange gamma correctly so that when we construct theta in this way then
    # gamma * theta == xi
    # Currently, gamma * log10(q(idxall)) == xi
    # So we need to know how to go from idxall to idxtheta

    idxconstrain = np.setdiff1d(idxall, idxvary)
    if idxconstrain.size != n_constrain:
        raise ValueError("Number of constraints does not match.")

    idxtheta = np.concatenate((idxconstrain, idxvary))
    # Note that theta == log10(q(idxtheta));
    mask = np.isin(idxtheta, idxall)
    if not np.all(mask):
        raise ValueError("Some elements in idxtheta were not found in idxall")
    ind_all_to_theta = _match(idxtheta, idxall)
    new_gamma = gamma[:, ind_all_to_theta]

    # Now do the qr factorization of new_gamma
    U, R = scipy.linalg.qr(new_gamma)
    R1 = R[:, :n_constrain]
    R2 = R[:, n_constrain:]

    return U, R1, R2, idxtheta, new_gamma, R, idxconstrain, n_constrain


def _sub2ind(shape, i, j):
    n_rows, n_cols = shape
    return n_cols * i + j


def _mst_path(csgraph, predecessors, start, end):
    """Construct a path along the minimum spanning tree"""
    preds = predecessors[start]
    path = [end]
    while end != start:
        end = preds[start, end]
        path.append(end)
    return path[::-1]


def constrain_rate(
    idxall, source_idx, target_idx=None, type="fix", constant=1.0
):
    """Create linear constraint on rate constants in the Q matrix

    This function returns a matrix, ``A``, of coefficients giving the linear
    constraints on specified reate constants in the Q matrix and also returns
    the equivalence column vector, ``B``, such that $A*\theta = B$ where
    $\theta$ is a column vector of the rate constants.

    Parameters
    ----------
    idxall : tuple
        Indices of all the rate constants that make up Q as a tuple of arrays,
        one for each dimension of Q. numpy.nonzero returns such a tuple.
    source_idx : tuple
        The indices of the source rate constants as a tuple of arrays. These are the
        rate constants that are free to vary.  When type is 'fix', source_idx
        is the indices of the rates that should be fixed to a certain value
    target_idx : 1-d array
        Indices of rate constants that will be constrained as a tuple of arrays
    type : str, {'fix', 'constrain', 'loop'}
        The type of constraint, either 'fix', 'constrain', or 'loop'
    constant : float, optional
        Vector of the constants for the constraints. ``constrain_rate`` will
        return the log10(c) in B.  If c is not given, it is assumed to be 1

    Returns
    -------
    A : 2-d array
        Coefficients giving the linear constraints on specified rate constants
        in the Q matrix
    B : 2-d array
        A column vector of the equivalences
    """

    num_rates = len(idxall[0])
    num_constraints = len(source_idx[0])
    if np.ndim(constant) == 0:
        # constant is a scalar
        constant = constant * np.ones(num_constraints)

    if type == "fix":
        A = np.zeros((num_constraints, num_rates))
        idx2 = _match_hash(zip(*idxall), zip(*source_idx))
        A[np.arange(num_constraints), idx2] = 1
        B = np.log10(constant).reshape(-1, 1)

        return A, B

    if type == "constrain":
        if len(target_idx[0]) != num_constraints:
            raise ValueError(
                "The number of source and target rates for constraint must be equal"
            )

        A = np.zeros((num_constraints, num_rates))
        idx2 = _match_hash(zip(*idxall), zip(*source_idx))
        idx3 = _match_hash(zip(*idxall), zip(*target_idx))
        A[np.arange(num_constrants), idx2] = -1
        A[np.arange(num_constrants), idx3] = 1
        B = np.log10(constant).reshape(-1, 1)

        return A, B

    if type == "loop":
        A = np.zeros((1, num_rates))
        idx2 = _match_hash(zip(*idxall), zip(*source_idx))
        idx3 = _match_hash(zip(*idxall), zip(*target_idx))

        idx = np.concatenate((idx2, idx3), axis=-1)
        A[0, idx] = np.concatenate(
            (np.ones(len(source_idx[0])), -1 * np.ones(len(target_idx[0])))
        )
        B = np.log10(constant).reshape(-1, 1)

        return A, B

    raise ValueError("`type` must be one of 'fit', 'constrain', or 'loop'")


def mr_constraints(
    q, idxall, idxconstrain=None, idxmr=None, gamma=None, xi=None
):
    """Find rates to fix for microscopic reversibility

    Parameters
    ----------
    q : 2-d array
        The Q matrix of transition probabilities
    idxall : array
        The indices of `q` that give all the rate constants
    idxconstrain : array
        The indices of `q` that are constrained
    idxmr : array
    gamma : 2-d array
        Set of constraints on the rate constants
    xi : 1-d array
        A column array of constants for the linear constraints

    Returns
    -------
    Gamma : 2-d array
    Xi : 2-d array
        A column vector of the equivalence values
    idxConstrain : 1-d array
        Indices of the rates to constrain
    idxMR : tuple of arrays
        Indices of the rates constrained by microscopic reversibility, as
        returned by `np.nonzero`
    MST : 2-d sparse array
        The minimum spanning tree

    Notes
    -----
    1) Create an undirected graph of Q (with the physical constrains from
       idxconstrain -- this may not be needed)
       a) Need to convert a directed graph to an undirected graph
           i) Matlab in the example in graphisomorphism uses G2 = G1 + G1';
           ii) One issue is that the rates from idxconstrain will be in either
           the upper or lower triangle of the Q matrix, but for an undirected
           graph, the Matlab algorithms ignore the upper triangle, so we'll
           need to be sure to set the weights for both transitions in the edge
           containing the rates in idxconstrain

    2) Assign weights to the edges in the graph G (which is really the
       constrained Q matrix [undirected])

       a) Do so in such a way that

          weight = 1 for edges (i.e. connections) to be included in the
          minimum spanning tree. These connections will be set by physical
          constraints (and not microscopic reversibility, if possible)

          weights = s where s is the # of states for edges that the user
          wants to be set by microscopic reversibility (and therefore these
          are to be excluded from the minimum spanning tree)

          weights = 2 for any other edge

    3) Find the minimum spanning tree and from this select rates to constrain
       by microscopic reversibility (the weights from step 2a) should make it so
       that rates the user wants set by MR are excluded from the MST if possible

    4) Find the unique shortest path (I assume this is guaranteed to exist in
       a minimum spanning tree) between each pair of states connected by an edge
       that is not in the MST

       a) These paths will determine which cycle to use for the constrain
    """

    num_rates = q.shape[0]

    if idxconstrain is not None and gamma is not None:
        num_constraints, num_rates_gamma = gamma.size
        gamma_rank = scipy.linalg.matrix_rank(gamma)
        if num_constraints > gamma_rank:
            warnings.warn(
                "The constraints on the rates in Q are not all independent"
            )
        if num_rates != num_rates_gamma:
            raise ValueError(
                "q had {} rates but gamma had {} rates".format(
                    num_rates, num_rates_gamma
                )
            )
    else:
        num_constraints = 0

    if idxmr is None:
        idxmr = []

    # According to a DeprecationWarning from numpy,
    # "Use `array.size > 0` to check that an array is not empty."
    if np.intersect1d(idxconstrain, idxmr).size < 1:
        raise ValueError(
            "Some constraints are set by physical constraints "
            "and microscopic reversibility"
        )

    # Check that all connections between vertices are bi-directional
    # This should be equivalent to testing if a boolean version of
    # the matrix is symmetric
    if not np.all(q.astype(bool) == q.astype(bool).T):
        raise ValueError(
            "Connections between states in q must be bi-directional"
        )

    # Force diagonal of the Q matrix to be zero for purpose of finding MST
    if np.count_nonzero(q.diagonal()) > 0:
        warnings.warn(
            "The diagonal elements of q were not all zero. "
            "They are being set to zero to find the minimum "
            "spanning tree."
        )
        Q[np.eye(*q.shape, dtype=bool)] = 0

    # scipy.sparse.minimum_spanning_tree goes through the nodes in the
    # graph rowwise (I believe this is due to the way indices are stored
    # using CSR sparse format). Hence, if the graph has entries
    # in both the upper and lower triangles of the matrix then it will
    # first find the entries in the upper triangle. Moreover, the weight
    # of the connection in the upper triangle will be used.
    G = np.zeros_like(q)
    G[q != 0] = 2
    G[idxconstrain] = 1
    G = np.minimum(G, G.T)

    # Assigning weights to connections the user wants fixed by microscopic
    # reversibility after the weights of connections fixed by physical
    # constraints could lead to non-independent constraints if some of the
    # physical constraints are on the connections fixed by MR. The weights of
    # rates fixed by MR could be set first to avoid this.
    n_states = q.shape[0]
    G[idxmr] = n_states
    G = np.maximum(G, G.T)

    # The graph should have the same weights in the upper and lower triangles
    # but just in case, let's take the lower triangular portion
    G = np.tril(G)

    MST = scipy.sparse.csgraph.minimum_spanning_tree(G)

    # Let's just use the transitions in the lower triangular part of Q to fix for MR
    # Thus the rates to fix for microscopic reversibility are in the lower
    # triangular of Q but not in the minimum spanning tree
    MR_initial_mask = np.tril(q.astype(bool)) != MST.astype(bool)

    # Check that the user did not specify a rate to fix for microscopic
    # reversibility that is in the upper triangular
    mr_mask = np.zeros_like(q, dtype=bool)
    mr_mask[idxmr] = True
    mr_mask = np.triu(mr_mask)
    MR_mask = (MR_initial_mask & ~mr_mask.T) | (mr_mask & MR_initial_mask.T)

    idxMR = np.nonzero(MR_mask)
    num_mr_constraints = idxMR[0].size

    # An alternative way, perhaps:
    # set(zip(idxMR[1], idxMR[0])).difference(zip(*idxmr))
    # now, we'd need the index of the overlap from idxMR ...

    distances, predecessors = scipy.sparse.csgraph.shortest_path(
        MST, directed=False, indices=(ii, jj), return_predecessors=True
    )

    Gamma = np.zeros((num_constrants + num_mr_constraints, num_rates))
    idxConstrain = np.zeros(num_constrants + num_mr_constraints)
    Xi = np.zeros((num_constrants + num_mr_constraints, 1))

    Gamma[:num_constrants, :] = gamma
    idxConstrain[:num_constrants, :] = idxconstrain
    Xi[:num_constrants, :] = xi

    for n, sub in enumerate(zip(*idxMR)):
        ii, jj = sub

        pth = _mst_path(MST, predecessors, ii, jj)
        pth_next = np.roll(pth, -1)
        src = _sub2ind(q.shape, pth, pth_next)

        rev_path = path[::-1]
        rev_path_next = np.roll(rev_path, -1)
        tgt = _sub2ind(q.shape, rev_path, rev_path_next)

        tmpg, tmpxi = constrain_rate(q, idxall, src, tgt, "loop")

        ind = num_constrants + n
        Gamma[ind, :] = tmpg
        Xi[ind, :] = tmpxi
        idxConstrain[ind, :] = _sub2ind(ii, jj)

    return Gamma, Xi, idxConstrain, idxMR, MST
