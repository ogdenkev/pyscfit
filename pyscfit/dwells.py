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
    unresolved_inds : 1-d array
        Indices of unresolved durations from the original list of dwell times
    """

    TOL = 1e-12

    resolved_mask = (dwells >= openres) & (states != 0)
    resolved_mask |= (dwells >= shutres) & (states == 0)
    unresolved_inds = np.nonzero(~resolved_mask)

    resolved_dwells = []
    resolved_states = []
    for resolved, sojourns in itertools.groupby(
        zip(dwells, states, resolved_mask), key=lambda tup: tup[2]
    ):
        if resolved:
            for d, s, r in sojourns:
                resolved_dwells.append(d)
                resolved_states.append(s)
        else:
            if resolved_dwells:
                # There has already been a resolved sojourn, so we can add
                # the unresolved time to the last resolved sojourn
                sojourn_times, sojourn_states, sojourn_resolved = zip(
                    *sojourns
                )
                unresolved_time = sum(sojourn_times)
                resolved_dwells[-1] += unresolved_time

    return resolved_dwells, resolved_states, unresolved_inds


def concatdwells():
    raise NotImplementedError


def find_tcrit():
    raise NotImplementedError


def dwt_read():
    raise NotImplementedError


def scan_read():
    raise NotImplementedError
