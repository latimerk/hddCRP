import numpy as np

# for typing and validating arguments
from numpy.typing import ArrayLike
from typing import Callable

# TODO: Simulate a set of sessions from a known Markov model
#       Each session's parameters change slightly (fixed within session)
#       Use to ask how well an hddCRP can fit each session's variables as a function of the amount of data (and alpha and distance parameter) 
    # TODO: Generate sequence of X order markov chain matrices
    # TODO: Simulate a time series from an X order markov chain
def simulate_markov_chain(initial_state_prob : ArrayLike, transition_matrix : ArrayLike, T : int) -> np.ndarray:
    initial_state_prob = np.array(initial_state_prob)
    if(initial_state_prob.ndim == 1):
        initial_state_prob = initial_state_prob[:,np.newaxis]
    assert initial_state_prob.ndim == 2, "initial_state_prob must have 1 or 2 dimensions"
    assert np.all(initial_state_prob > 0), "initial probability must be positive"
    initial_state_prob = initial_state_prob / np.sum(initial_state_prob, axis=0)
    M = np.size(initial_state_prob)
    order = initial_state_prob.shape[1]
    assert order > 0, "order must be positive"
    assert M > 0, "number of states must be positive"

    T = int(T)
    assert T > 0, "number of steps must be positive"

    transition_matrix = np.array(transition_matrix)
    assert np.all(np.equal(transition_matrix.shape, M)), "transition_matrix.shape[:] must be equal to initial_state_prob.shape[0]"
    assert transition_matrix.ndim == order+1, "transition_matrix.ndim must be equal to (initial_state_prob.shape[1]): the order of the chain"

    c = np.sum(transition_matrix, axis=-1)[...,np.newaxis];
    transition_matrix = transition_matrix / c

    states = range(M)

    Y = np.zeros((T),dtype=int)
    for tt in range(order):
        Y[tt] = np.random.choice(states, p=initial_state_prob[:,tt])
    for tt in range(order,T):
        p = transition_matrix[np.s_[tuple(Y[tt-order:tt])], :].flatten();
        Y[tt] = np.random.choice(states, p=p)

    return Y




# TODO: Generate a hddCRP with specific alpha and distances
#  Use to see how well we can recover distance parameter and alphas