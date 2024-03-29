"""Functions for processing dwell times"""

import itertools

import numpy as np
import scipy.linalg

from .qmatrix import equilibrium_occupancy


def impose_res(dwells, states, openres, shutres, return_unresolved_inds=False):
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

    resolved_dwells = np.array(resolved_dwells)
    resolved_states = np.array(resolved_states)

    if return_unresolved_inds:
        return resolved_dwells, resolved_states, unresolved_inds
    else:
        return resolved_dwells, resolved_states


def fix_shut_amps(dwells, states, zero_amp=0.0):
    """Force shut state amplitudes to be zero
    
    Parameters
    ----------
    dwells : 1-d array
        The duration of sojourns in the states given by the input array states
    states : 1-d array
        Identity of the state of each dwell time
    zero_amp : fload, optional
        A threshold amplitude for considering a dwell to be a closed state.
        If `np.abs(states[i]) < zero_amp` is true, then the amplitude will
        be forced to zero.
    
    Returns
    -------
    zero_amp_dwells : 1-d array
    zero_amp_states : 1-d array
    """

    zero_amp_dwells = []
    zero_amp_states = []

    for is_shut, sojourn in itertools.groupby(
        zip(dwells, states), key=lambda tup: np.abs(tup[0]) < zero_amp
    ):
        sjrn_dwells, sjrn_states = zip(*sojourn)

        if is_shut:
            zero_amp_dwells.append(sum(sjrn_dwells))
            zero_amp_states.append(0)
            continue

        zero_amp_dwells.extend(sjrn_dwells)
        zero_amp_states.extend(sjrn_states)

    return np.array(zero_amp_dwells), np.array(zero_amp_states)


def concat_dwells(dwells, states, tol=0.1, mode="first"):
    """Concatenate contiguous open and closed durations
    
    Parameters
    ----------
    dwells : 1-d array
    states : 1-d array
    tol : float, optional
        The difference in picoamperes beyond which two adjacent states
        are considered to be distinct. Only adjacent dwells whose amplitudes
        differ by strictly less than `tol` will be combined.
    mode : {'first', 'mean'}, optional
        The strategy to combine amplitudes when dwells are concatenated.
        If mode is 'first' (the default), then only the amplitude of the
        first event is kept; other amplitudes are discarded. If 'mean', then
        the average amplitude from all concatenated events is kept.
    
    Returns
    -------
    concatenated_dwells : 1-d array
        The concatenated dwells
    concatenated_states : 1-d array
        The concatenated states
    """

    n_states = dwells.size
    if states.size != n_states:
        raise ValueError(
            "dwells and states must have the same number of sojourns, "
            "but found {} sojourns in dwells and {} in states".format(
                n_states, states.size
            )
        )

    delta_amps = np.ediff1d(states)
    concatenated_dwells = []
    concatenated_states = []

    current_dwell = dwells[0]
    current_state = states[0]
    n_concat = 1

    for i, d_amp in enumerate(delta_amps):
        concat = np.abs(d_amp) < tol
        if concat:
            current_dwell += dwells[i + 1]
            if mode == "mean":
                current_state += d_amp
                n_concat += 1
        else:
            concatenated_dwells.append(current_dwell)
            concatenated_states.append(current_state / n_concat)
            current_dwell = dwells[i + 1]
            current_state = states[i + 1]
            n_concat = 1

    concatenated_dwells.append(current_dwell)
    concatenated_states.append(current_state / n_concat)

    return np.array(concatenated_dwells), np.array(concatenated_states)


def find_tcrit():
    raise NotImplementedError


def dwt_read():
    raise NotImplementedError


def scan_read():
    raise NotImplementedError


def monte_carlo_dwells(q, A, F, n, ini_state=None, seed=None):
    """Simulate a continuous time Markov process governed by Q matrix
    
    A sequence of dwell times is produced in Monte Carlo fashion
    
    Parameters
    ----------
    q : 2-d array
        Q matrix -- cannot vary with time
    A : 1-d array
        Array of indices of the open (or up) state
    F : 1-d array
        Array of indices of the closed (or down) state
    n : int
        The number of transitions to simulate
    ini_state : int, optional
        The initial state of the system
    seed : {None, int, array_like[ints], ISeedSequence,
    BitGenerator, Generator}, optional
        Seed to use with numpy.random.default_rng()
    
    Returns
    -------
    dwells : 1-d array
        The simulated dwell times
    states : 1-d array
        The states corresponding to the dwell times, values of
        0 = shut (or down) and values of 1 = open (or up)
    """

    n_states = q.shape[0]
    batch_size = (n_states * 2 * n, n)

    if (A.size + F.size) != n_states:
        raise ValueError(
            "Q had {} states but only {} states found in A and F".format(
                n_states, A.size + F.size
            )
        )

    rg = np.random.default_rng(seed)

    def get_random_vals(generator=rg, q=q, batch_size=batch_size):
        norm_vals = generator.random(size=batch_size[0])
        ind_norm = 0
        exp_vals = [
            generator.exponential(scale=-1 / v, size=batch_size[1])
            for v in np.diagonal(q)
        ]
        ind_exp = [0] * np.diagonal(q).size

        return norm_vals, ind_norm, exp_vals, ind_exp

    die_rolls, ind_roll, random_dwell_times, ind_rand_dwell = get_random_vals()

    dwells = np.zeros(n)
    states = np.full(n, np.nan)

    if ini_state is None:
        p0 = equilibrium_occupancy(q)
        die = die_rolls[ind_roll]
        ind_roll += 1
        state = np.nonzero(die <= np.cumsum(p0))[0][0]
    else:
        state = ini_state

    prob_t = q.copy()
    prob_t[np.eye(n_states, dtype=np.bool_)] = 0
    prob_t = np.cumsum(prob_t / prob_t.sum(axis=1, keepdims=True), axis=1)

    if state in A:
        current_class = A
        current_amplitude = 1
    elif state in F:
        current_class = F
        current_amplitude = 0
    else:
        raise ValueError("Current state not in A or F")

    for ii in range(n):
        while state in current_class:
            time = random_dwell_times[state][ind_rand_dwell[state]]
            ind_rand_dwell[state] += 1
            dwells[ii] += time

            pt = prob_t[state]

            die = die_rolls[ind_roll]
            ind_roll += 1

            for ind, p in enumerate(pt):
                if ind == state:
                    continue
                if die <= p:
                    break

            state = ind

            if (
                ind_roll >= batch_size[0]
                or ind_rand_dwell[state] >= batch_size[1]
            ):
                die_rolls, ind_roll, random_dwell_times, ind_rand_dwell = (
                    get_random_vals()
                )

        states[ii] = current_amplitude
        current_amplitude ^= 1
        current_class = A if state in A else F

    return dwells, states
