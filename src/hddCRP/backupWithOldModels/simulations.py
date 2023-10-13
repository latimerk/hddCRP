import numpy as np
from collections import defaultdict


# for typing and validating arguments
from numpy.typing import ArrayLike
from typing import Callable

from hddCRP.behaviorDataHandlers import create_distance_matrix, parameter_vectorizer_for_distance_matrix
from hddCRP.behaviorDataHandlers import distance_function_for_maze_task
from hddCRP.modelFittingSequential import sequentialhddCRPModel



# TODO: Simulate a set of sessions from a known Markov model
#       Each session's parameters change slightly (fixed within session)
#       Use to ask how well an hddCRP can fit each session's variables as a function of the amount of data (and alpha and distance parameter) 
    # TODO: Generate sequence of X order markov chain matrices
    # TODO: Simulate a time series from an X order markov chain
def simulate_markov_chain(initial_state_prob : ArrayLike, transition_matrix : ArrayLike, T : int, rng : np.random.Generator) -> np.ndarray:
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
        Y[tt] = rng.choice(states, p=initial_state_prob[:,tt])
    for tt in range(order,T):
        p = transition_matrix[np.s_[tuple(Y[tt-order:tt])], :].flatten();
        Y[tt] = rng.choice(states, p=p)

    return Y


# TODO: Generate a hddCRP with specific alpha and distances
#  Use to see how well we can recover distance parameter and alphas
#  Sequential CRP: that way we can model time series data where the hierarchical groups depend on recent context (n-th order markov model)
def simulate_sequential_hddCRP(session_length : int | ArrayLike, session_types : ArrayLike, symbols : ArrayLike | int,
                               depth : int, rng : np.random.Generator, alphas : ArrayLike | float, between_session_time_constants : ArrayLike, nback_scales : ArrayLike = None, # between_session_scales : ArrayLike = 1,
                               within_session_time_constant : float | ArrayLike = np.inf, 
                               base_measure : ArrayLike | None = None):
    session_types = np.array(session_types).flatten()
    num_sessions = session_types.size
    assert num_sessions > 0, "no sessions given"

    session_labels = np.unique(session_types);
    num_session_labels = session_labels.size

    if(np.isscalar(session_length)):
        session_length = np.ones((num_sessions), dtype=int)* int(session_length)
    else:
        session_length = np.array(session_length, dtype=int).flatten()
    assert np.all(session_length > 0), "session lengths must be positive integers"
    assert session_length.size == num_sessions, "session_length and session_types must have the same size"

    if(isinstance(symbols, int) and symbols > 0):
        symbols = np.arange(symbols)
    else:
        symbols = np.array(symbols).flatten()
    M = symbols.size
    assert M > 0, "no symbols given"

    depth = int(depth)
    assert depth > 0, "depth must be positive integer"
    if(np.isscalar(alphas)):
        alphas = np.ones((depth))*alphas
    else:
        alphas = np.array(alphas).flatten()
    assert alphas.size == depth, "alphas must be of length 'depth' or a scalar"

    between_session_time_constants = np.array(between_session_time_constants)
    assert between_session_time_constants.shape == (num_session_labels, num_session_labels), "between_session_time_constants must be of size (K,K) where K = np.unique(session_labels).size"
    assert np.all(between_session_time_constants > 0), "between_session_time_constants must be positive"

    if(nback_scales is None):
        nback_scales = np.ones((depth-1))
    elif(np.isscalar(nback_scales)):
        nback_scales = np.ones((depth-1)) * nback_scales;
    else:
        nback_scales = np.array(nback_scales)
        
    assert nback_scales.shape == (depth-1,), "nback_scales must be a scalar or size (depth-1)"
    

    if(np.isscalar(within_session_time_constant)):
        distinct_within_session_distance_params = False
        within_session_time_constant = np.ones((num_session_labels))*within_session_time_constant
    else:
        distinct_within_session_distance_params = True
        within_session_time_constant = np.array(within_session_time_constant).flatten()
    assert within_session_time_constant.size == num_session_labels and np.all(within_session_time_constant > 0), "within_session_time_constant must be positive scalar or size (K) where K = np.unique(session_labels).size"

    if(base_measure is None or np.isscalar(base_measure)):
        base_measure = np.ones((M))
    else:
        base_measure = np.array(M).flatten()
    base_measure = base_measure / np.sum(base_measure)
    assert np.all(base_measure > 0) and base_measure.size == M, "base_measure must be positive array of length M = {symbols.size if symbols is Array, or symbols if symbols is scalar int}"

    seqs = [np.zeros((tt),dtype=symbols.dtype) for tt in session_length];

    D, DB, variable_names, num_timescales = create_distance_matrix(seqs, session_types, 
                        distinct_within_session_distance_params=distinct_within_session_distance_params,
                        nback_scales=depth-1, actions=symbols);
    D[:,:,2:] = 0;
    DB[...] = 0


    ## set up parameters in a specific vectorized form
    param_names, params_vector, is_within_timescale, is_between_timescale, is_nback_scale = parameter_vectorizer_for_distance_matrix(variable_names, session_types, within_session_time_constant = within_session_time_constant, between_session_time_constants = between_session_time_constants, nback_scales = nback_scales);
    param_types = {"is_within_timescale" : is_within_timescale,
                   "is_between_timescale" : is_between_timescale,
                   "is_nback_scale" : is_nback_scale}

    ##
    log_params = np.log(params_vector);


    T_total = D.shape[0];
    F = np.zeros((T_total, T_total, depth))
    B = np.zeros((T_total, M))
    C_y = np.zeros((T_total),dtype=int)
    C_y[:] = -1
    C_depth = np.zeros((T_total),dtype=int)
    C_ctx = np.zeros((T_total, depth),dtype=int)
    C_ctx.fill(-1)

    tt_ctr = 0;

    for ss in range(num_sessions):
        for tt in range(session_length[ss]):
            for dd in range(depth):
                # set contexts
                if(tt >= dd):
                    C_ctx[tt_ctr, dd] = C_y[tt_ctr - dd]

                    if(dd > 0):
                        D[tt_ctr, C_y == C_y[tt_ctr - dd], 1+dd] = 1;
                        DB[tt_ctr,C_y[tt_ctr - dd],dd-1] = 1
                else:
                    C_ctx[tt_ctr, dd] = -2
            D_c = D[[tt_ctr],:tt_ctr,...];
            DB_c = DB[[tt_ctr], ...];
            F_c, B_c = distance_function_for_maze_task(log_params, D_c, DB_c, depth,  num_timescales)
            # print("D_c " + str(D_c.shape))
            # print("DB_c " + str(DB_c.shape))
            # print("F_c " + str(F_c.shape))
            # print("B_c " + str(B_c.shape))
            F[[tt_ctr],:tt_ctr,:] = F_c
            B[[tt_ctr],:] = B_c 
            weighted_base_measure = base_measure * B[tt_ctr,...]
            alphas_c = alphas.copy();
            alphas_c[0] *= weighted_base_measure.sum()
            weighted_base_measure /= weighted_base_measure.sum()

            for dd in range(depth-1,-1,-1):


                if(not np.isinf(alphas_c[dd])):
                    # find all previous nodes with matching context at current depth
                    nodes_with_same_context = np.where(np.all(C_ctx[:(tt_ctr),:(dd+1)] == C_ctx[tt_ctr,:(dd+1)], axis=1))[0]
                    
                    # select previous node with appropriate weight
                    wts = F[tt_ctr, nodes_with_same_context,dd]
                    Ys = C_y[nodes_with_same_context];

                    
                    ms = np.array([wts[Ys == mm].sum() for mm in range(M)] + [alphas_c[dd]])
                    ms = ms/np.sum(ms);

                    rc = rng.choice(M+1, p=ms);
                else:
                    rc = M;

                # if jump to shallower context
                if(rc < M):
                    C_y[tt_ctr] = rc;
                    C_depth[tt_ctr] = dd;
                    break;
                elif(dd == 0):

                    C_y[tt_ctr] = rng.choice(M, p=weighted_base_measure);
                    C_depth[tt_ctr] = dd;
                    break;

            # store observation 
            seqs[ss][tt] = symbols[C_y[tt_ctr]]
            tt_ctr += 1
    C = {'C_y' : C_y, 'F' : F, 'D' : D, 'B' : B, 'DB' : DB,'C_ctx' : C_ctx,
         'session_lengths' : session_length, "symbols" : symbols, "alphas" : alphas, "base_measure" : base_measure, 
         "param_vector" : params_vector, "param_names" : param_names, "variable_names" : variable_names, "param_types" : param_types, "C_depth" : C_depth}
    return (seqs, C)
