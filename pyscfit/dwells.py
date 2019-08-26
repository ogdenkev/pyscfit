"""Functions for processing dwell times"""

import numpy as np
import scipy.linalg

def imposeres(dwells, states, openres, shutres):
    """Impose resolutions for sojourns in an idealized channel recording

    Parameters
    ----------
    dwells : 1-d array
        The duration of sojourns in the states given by the input array states
    states : 1-d array
        Identity of the state of each dwell time
    openres : float
        Resolution (in milliseconds) of durations in states other than 0
    shutres : float
        Shut time resolution (in ms) of durations in state 0

    Returns
    -------
    resolved_dwells : 1-d array
        Resolved durations
    resolved_states : 1-d array
        States corresponding to the resolved durations
    unresolved : 1-d array
        Indices of unresolved durations from the original list of dwell times
    """

    TOL = 1e-12
    # Make sure we start with a resolvable dwell
    start = 0
    while (dwells[start] + tol < openres & states[start] != 0) || (dwells(ii)+tol<shutres & states(ii)==0)
        dwells(ii)=[];
        states(ii)=[];
        if isempty(dwells)
            break

    raise NotImplementedError


def concatdwells():
    raise NotImplementedError


def find_tcrit():
    raise NotImplementedError


def dwt_read():
    raise NotImplementedError


def scan_read():
    raise NotImplementedError
