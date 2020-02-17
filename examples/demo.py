"""Single channel rate estimation

Simulate single channel dwell times and estimate rate constants using
maximum likelihood of the full idealized record with an exact
correction for missed events
"""

import numpy as np
import scipy.linalg
import joblib

import pyscfit.dwells
import pyscfit.qmatrix
import pyscfit.fitting

# Gating mechanism from Q-Matrix Cookbook chapter of Single Channel Recording
n_states = 5
q = np.zeros((n_states, n_states))
idx_vary = tuple(
    zip((3, 0), (0, 1), (2, 1), (1, 2), (3, 2), (0, 3), (2, 3), (4, 3), (3, 4))
)
q[idx_vary] = np.array([15, 50, 15e3, 500, 50, 3e3, 4e3, 10, 2e3])

# Microscopic reversibility
idx_constrain = ((1,), (0,))
q[idx_constrain] = (
    q[[0, 1, 2, 3], [1, 2, 3, 0]].prod() / q[[0, 3, 2], [3, 2, 1]].prod()
)

idx_all = (idx_vary[0] + idx_constrain[0], idx_vary[1] + idx_constrain[1])

q -= np.diag(q.sum(axis=1))
q *= 1e-3

# Open states
A = np.arange(2)

# Shut states
F = np.arange(2, n_states)

# Simulate data
print("Simulating data ...")
n_dwells = 50000
dwells, states = pyscfit.dwells.monte_carlo_dwells(q, A, F, n_dwells)

# Resolution (aka dead time), in milliseconds
td = 0.1
print("Imposing resolution of {} milliseconds".format(td))
resolved_dwells, resolved_states = pyscfit.dwells.impose_res(
    dwells, states, td, td
)

print("Concatenating adjacent dwells in the same state ...")
concat_dwells, concat_states = pyscfit.dwells.concat_dwells(
    resolved_dwells, resolved_states
)

q_guess = np.zeros((n_states, n_states))
# (ind1, ind2, rate in s^-1)
rates = [
    (0, 3, 1e4),
    (3, 0, 1e3),
    (0, 1, 5e7 * 1e-7),
    (1, 0, 0.0),
    (1, 2, 100),
    (2, 1, 3e4),
    (2, 3, 1e3),
    (3, 2, 5e7 * 1e-7),
    (3, 4, 1e3),
    (4, 3, 1e7 * 1e-7),
]
for i, j, rate in rates:
    q_guess[i, j] = rate * 1e-3

src = ([0, 1, 2, 3], [1, 2, 3, 0])
tgt = ([0, 3, 2, 1], [3, 2, 1, 0])
gamma, xi = pyscfit.fitting.constrain_rate(idx_all, src, tgt, type="loop")

print("Finding rate constants ...")
rates, ll, qnew, hess, cov, cor, hist = pyscfit.fitting.fit_rates(
    concat_dwells, q_guess, A, F, td, idx_all, idx_vary, gamma, xi
)

outfile = "demo_results.joblib"
print("Saving results to {}".format(outfile))
with open(outfile, "wb") as fd:
    joblib.dump(
        {
            "rates": rates,
            "ll": ll,
            "qnew": qnew,
            "hess": hess,
            "cov": cov,
            "cor": cor,
            "hist": hist,
        },
        fd
    )

print("Fitted rates:")
print(rates)
