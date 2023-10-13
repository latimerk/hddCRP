import numpy as np
import networkx as nx
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl

# for typing and validating arguments
from numpy.typing import ArrayLike
from typing import Callable

from hddCRP.behaviorDataHandlers import create_distance_matrix, parameter_vectorizer_for_distance_matrix
from hddCRP.behaviorDataHandlers import create_context_tree
from hddCRP.modelFitting import exponential_distance_function, exponential_distance_function_for_maze_task, hddCRPModel, complete_exponential_distance_function_for_maze_task

def node_name(node, layer):
    return "l" + str(layer) + "_" + str(node);

def graph_connections(session_lengths, C_ptr, ww = 2, ws=8, hs = 5, hh=0.1):
    depth = C_ptr.shape[1]

    session_lengths = np.array(session_lengths, dtype=int)

    session_idxs = np.append(0, session_lengths.cumsum())
    assert (session_idxs[-1] <= C_ptr.shape[0]), "C_ptr and session_lengths do not match"
    nodes = session_idxs[-1];#C_ptr.shape[0]

    # setup nodes
    pos = defaultdict(lambda: (0,0))
    G = nx.DiGraph()
    for dd in range(depth):
        for nn in range(nodes):
            sess_num = np.where(session_idxs > nn)[0][0]-1

            cc = node_name(nn,dd);
            G.add_node(cc)

            session_t = (nn-session_idxs[sess_num])/(session_lengths[sess_num] - 1)*2 - 1


            pos[cc] = [nn*ww + sess_num*ws, (depth-dd-1)*hs + hh*(session_t**2)]
    for dd in range(depth):
        for nn in range(nodes):
            target_dd = dd;
            target_nn = C_ptr[nn,dd]
            if(dd > 0 and target_nn == nn):
                target_dd -= 1;
            G.add_edges_from([(node_name(nn,dd), node_name(target_nn,target_dd))])
    return (G, pos)

def make_graph_plot(connection_data, ax, hs=15,hh=5,ws=8, ww=4,num_sessions=None):
    if(num_sessions is None):
        num_sessions = len(connection_data['session_lengths'])

    depth = connection_data['C_ptr'].shape[1]
    M = np.unique(connection_data['C_y']).size
    T = np.sum(connection_data['session_lengths'][:num_sessions])
    G, pos = graph_connections(connection_data['session_lengths'][:num_sessions], connection_data['C_ptr'], hs=hs,hh=hh, ww=4,ws=8)
    node_size = 30;


    edge_color = connection_data["C_ctx"][:T,:]
    bb = np.arange(depth)*(M+1);
    edge_color += bb[np.newaxis,:]
    edge_color = edge_color.flatten(order='F')

    node_color=connection_data['C_y'][:T,:].flatten(order='F')


    nodes = nx.draw_networkx_nodes(G, pos,   node_size=node_size, node_color=node_color)



    edges = nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=10,
        width=2,
        node_size=node_size,
        edge_color=edge_color
    )

    ax.set_axis_off()

    pc = mpl.collections.PatchCollection(edges)
            


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
                               depth : int, rng : np.random.Generator, alphas : ArrayLike | float, between_session_time_constants : ArrayLike, between_session_constant_scales : ArrayLike = None, # between_session_scales : ArrayLike = 1,
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

    if(not between_session_constant_scales is None):
        if(np.isscalar(between_session_constant_scales)):
            between_session_constant_scales = np.ones_like(between_session_time_constants) * between_session_constant_scales;
        assert between_session_constant_scales.shape == between_session_time_constants.shape, "between_session_constant_scales must be a scalar or size (K,K) where K = np.unique(session_labels).size"
    

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

    D, variable_names = create_distance_matrix(seqs, session_types, distinct_within_session_distance_params = distinct_within_session_distance_params,
                                                sequential_within_session_distances = True,
                                                sequential_between_session_same_block_distances = True,
                                                sequential_between_session_different_block_distances = True,
                                                within_block_distance_in_total_sessions  = True);


    D_nonsequential, _ = create_distance_matrix(seqs, session_types, distinct_within_session_distance_params = distinct_within_session_distance_params,
                                                sequential_within_session_distances = False,
                                                sequential_between_session_same_block_distances = False,
                                                sequential_between_session_different_block_distances = True,
                                                within_block_distance_in_total_sessions  = True);
    
    ## set up parameters in a specific vectorized form
    param_names, params_vector, is_within_timescale, is_between_timescale, is_between_constant_scale, timescale_inds, constant_scale_inds = parameter_vectorizer_for_distance_matrix(variable_names, session_types, within_session_time_constant = within_session_time_constant, between_session_time_constants = between_session_time_constants, between_session_constant_scales = between_session_constant_scales)
    param_types = {"is_within_timescale" : is_within_timescale, "is_between_timescale" : is_between_timescale, "is_between_constant_scale" : is_between_constant_scale,
                   "timescale_inds" : timescale_inds, "constant_scale_inds" : constant_scale_inds}

    ##
    log_params = np.log(params_vector);

    # F = np.apply_along_axis(lambda rr : exponential_distance_function_for_maze_task(rr, log_params), 2, D)
    D2 = np.min(D, axis=2)
    inds = np.argmin(D, axis=2)
    inds[np.isinf(D2)] = -1

    F = complete_exponential_distance_function_for_maze_task(D2, log_params, inds, timescale_inds, constant_scale_inds)
    
    np.fill_diagonal(F, 0);

    T_total = F.shape[0];
    C_ptr = np.zeros((T_total, depth),dtype=int)
    C_y   = np.zeros((T_total, depth),dtype=int)
    C_ctx = np.zeros((T_total, depth),dtype=int)

    C_ptr.fill(-1)
    C_y.fill(-1)
    C_ctx.fill(-1)

    tt_ctr = 0;

    for ss in range(num_sessions):
        for tt in range(session_length[ss]):
            for dd in range(depth):
                # set contexts
                if(tt >= dd):
                    C_ctx[tt_ctr, dd] = C_y[tt_ctr - dd, -1]
                else:
                    C_ctx[tt_ctr, dd] = -2

                # find all previous nodes with matching context at current depth
                nodes_with_same_context = np.where(np.all(C_ctx[:(tt_ctr+1),:(dd+1)] == C_ctx[tt_ctr,:(dd+1)], axis=1))[0]
                
                # select previous node with appropriate weight
                wts = F[tt_ctr, nodes_with_same_context]
                wts[nodes_with_same_context == tt_ctr] = alphas[dd];
                wts = wts/np.sum(wts)

                C_ptr[tt_ctr, dd] = rng.choice(nodes_with_same_context, p=wts);

                # set Y to connected node's Y
                if(C_ptr[tt_ctr, dd] != tt_ctr):
                    C_y[tt_ctr, dd] = C_y[C_ptr[tt_ctr, dd], dd]
                elif(dd > 0):
                    C_y[tt_ctr, dd] = C_y[tt_ctr, dd-1]
                else:
                    # if self connect: then draw Y from base measure
                    C_y[tt_ctr, dd] = rng.choice(M, p=base_measure)
            # store observation from final layer
            seqs[ss][tt] = symbols[C_y[tt_ctr, -1]]
            tt_ctr += 1
    C = {'C_y' : C_y, 'F' : F, 'D' : D, 'C_ptr' : C_ptr, 'C_ctx' : C_ctx, "D_nonsequential" : D_nonsequential,
         'session_lengths' : session_length, "symbols" : symbols, "alphas" : alphas, "base_measure" : base_measure, 
         "param_vector" : params_vector, "param_names" : param_names, "variable_names" : variable_names, "param_types" : param_types}
    return (seqs, C)

def create_hddCRPModel_from_simulated_sequential_hddCRP(seqs, C, rng : np.random.Generator = None, use_real_connections=True, use_nonsequential_filter_model=False):
    depth = C["alphas"].size
    Y = np.concatenate([np.array(ss).flatten() for ss in seqs], axis=0)
    block_ends = np.cumsum(np.array([np.size(ss) for ss in seqs],dtype=int))-1
    groupings = create_context_tree(seqs, depth=depth)

    if(use_nonsequential_filter_model):
        D_0 = C["D_nonsequential"]
    else:
        D_0 = C["D"];
    D = np.min(D_0, axis=2)
    inds = np.argmin(D_0, axis=2)
    inds[np.isinf(D)] = -1
    weight_func = lambda xx,yy : exponential_distance_function_for_maze_task(xx,yy)
    constant_scale_inds = C["param_types"]["constant_scale_inds"]
    timescale_inds = C["param_types"]["timescale_inds"]
    complete_weight_func = lambda d, log_timescales : complete_exponential_distance_function_for_maze_task(d, log_timescales, inds, timescale_inds, constant_scale_inds)

    if(use_real_connections):
        weight_params = np.log(C["param_vector"])
        alphas = C["alphas"]
    else:
        weight_params = np.zeros_like(C["param_vector"])
        weight_params[0] = np.log(rng.gamma(shape=2, scale=20, size=(1)))
        weight_params[1:] = np.log(rng.gamma(shape=2, scale=10,  size=(len(weight_params)-1)))

        alphas = rng.gamma(shape=2, scale=10, size=(depth))

    model = hddCRPModel(Y, groupings,
                        alphas,
                        D,
                        rng=rng,
                        Y_values=C["symbols"],
                        BaseMeasure=C["base_measure"],
                        weight_params=weight_params, 
                        weight_func=None, 
                        complete_weight_func=complete_weight_func, 
                        weight_param_labels=C["param_names"])
    model._param_types = C["param_types"]
    model._weight_function_setup = {"inds":inds,
                                    "timescale_inds":timescale_inds,
                                    "constant_scale_inds":constant_scale_inds}
    model._block_ends = block_ends
    if(use_real_connections):
        model._blank_connection_variables()
        model._C_ptr[:,:] = C["C_ptr"]
        model._initialize_predecessors();
        model._initialize_table_labels();
        model._initialize_table_cycles();
        model._initialize_table_label_counts();

    return model