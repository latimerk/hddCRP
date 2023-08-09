import numpy as np
from numpy.typing import ArrayLike
from typing import Callable
from __future__ import annotations

from hddCRP.modelFitting import hddCRPModel


EMPTY_INDICATOR = "NULL"




# TODO: Function to get tree of groups for time series of actions
def create_context_tree_single_session(seq : ArrayLike, depth : int = 3, delim : str = "-") -> np.ndarray:
    depth = int(depth)
    assert depth >= 1, "depth must be positive"
    
    seq = np.array(seq).flatten().astype(str);
    N = len(seq);
    assert N > 0, "seq can't be empty"

    grps = [['' for nn in range(N)] for dd in range(depth)]
    for dd in range(depth):
        if(dd > 0):
            for dd2 in range(dd,depth):
                grps[dd2] = [xx + delim + yy for xx,yy in zip(seq,grps[dd2])]
        seq =  np.roll(seq,1)
        seq[0] = EMPTY_INDICATOR
    grps = np.array(grps).T
    return grps
def create_context_tree(seqs : list[ArrayLike], depth : int = 3, delim : str = "-") -> list[np.ndarray]:
    return np.concatenate([create_context_tree_single_session(ss, depth=depth, delim=delim) for ss in seqs], axis=0)

# TODO: Function to get distances from a set of time series actions (set of sessions)
def create_distance_matrix(seqs : list[ArrayLike], block_ids : ArrayLike,   distinct_within_session_distance_params : bool = False,
                                                                            sequential_within_session_distances : bool = False,
                                                                            sequential_between_session_same_block_distances : bool = False,
                                                                            sequential_between_session_different_block_distances : bool = True,
                                                                            within_block_distance_in_total_sessions : bool = True,
                                                                            between_session_time_constants : ArrayLike = None, within_session_time_constant : float | ArrayLike = None):
    block_ids = np.array(block_ids).flatten()
    assert len(seqs) == len(block_ids), "number of sequences and block_ids must be the same (number of trials)"

    # num_sessions = len(seqs)
    session_lengths = [np.size(xx) for xx in seqs];
    session_start_indexes = np.append(0,np.cumsum(session_lengths));
    total_observations = np.sum(session_lengths);

    block_types = np.unique(block_ids);
    session_block_indexes = [np.where(block_ids == bb)[0] for bb in block_types];
    B = len(block_types)

    ##
    num_possible_parameters = (B ** 2) + (B if distinct_within_session_distance_params else 1);
    D = np.zeros((total_observations, total_observations, num_possible_parameters))
    D.fill(np.inf)
    param_ctr = 0;
    param_names = [];

    if(not (between_session_time_constants is None or within_session_time_constant is None)):
        between_session_time_constants = np.array(between_session_time_constants)
        assert between_session_time_constants.shape == (B, B), "between_session_time_constants must be of size (B,B) where B = np.unique(block_ids).size"
        assert np.all(between_session_time_constants > 0), "between_session_time_constants must be positive"

        if(np.isscalar(within_session_time_constant)):
            within_session_time_constant = np.ones((B))*within_session_time_constant
        else:
            within_session_time_constant = np.array(within_session_time_constant).flatten()
        assert within_session_time_constant.size == B and np.all(within_session_time_constant > 0), "within_session_time_constant must be positive scalar or size (B) where B = np.unique(block_ids).size"

        params = np.zeros((num_possible_parameters))
        vectorize_parameters = True;
    else:
        vectorize_parameters = False;

    ## Within session effects: should be sequential or all?
    for bb in range(B):
        for ss in session_block_indexes[bb]:
            t_end = session_start_indexes[ss+1]
            t_start = session_start_indexes[ss]
            t = t_end - t_start;
            r = np.arange(t,dtype=float)
            L = r[:,np.newaxis] - r; #np. np.ones((t, t));
            if(sequential_within_session_distances):
                # L = np.tril(L);
                L[L < 0] = np.inf

            D[t_start:t_end,t_start:t_end,param_ctr] = L
        if(vectorize_parameters):
            params[param_ctr] = within_session_time_constant[bb]
        if(distinct_within_session_distance_params):
            param_ctr += 1;
            param_names += ["within session - " + str(block_types[bb]) + " (units: actions)"]
    if(not distinct_within_session_distance_params):
        param_ctr += 1;
        param_names += ["within session - all (units: actions)"]

    ## Within block effects: distance is number of sessions between 
    for bb in range(B):
        for start_num, ss_start in enumerate(session_block_indexes[bb]):
            for end_num, ss_end in enumerate(session_block_indexes[bb]):
                distance_in_same_block = start_num - end_num; # units: sessions
                distance_all_sessions =  ss_start - ss_end; # units: sessions
                if(ss_start > ss_end or not sequential_between_session_same_block_distances):
                    t1_end = session_start_indexes[ss_start+1]
                    t1_start = session_start_indexes[ss_start]
                    t2_end = session_start_indexes[ss_end+1]
                    t2_start = session_start_indexes[ss_end]

                    D[t1_start:t1_end, t2_start:t2_end, param_ctr] = distance_all_sessions if within_block_distance_in_total_sessions else distance_in_same_block;
        if(vectorize_parameters):
            params[param_ctr] = between_session_time_constants[bb,bb]
        param_ctr += 1;
        param_names += ["within block - " + str(block_types[bb]) + " (units: sessions)"]


    for bb_start in range(B):
        for bb_end in range(B):
            if(bb_start == bb_end):
                continue;

            for ss_start in session_block_indexes[bb_start]:
                for ss_end in session_block_indexes[bb_end]:
                    if(ss_start > ss_end or not sequential_between_session_different_block_distances):
                        t1_end = session_start_indexes[ss_start+1]
                        t1_start = session_start_indexes[ss_start]
                        t2_end   = session_start_indexes[ss_end+1]
                        t2_start = session_start_indexes[ss_end]
                        D[t1_start:t1_end, t2_start:t2_end, param_ctr] = ss_start - ss_end;
            if(vectorize_parameters):
                params[param_ctr] = between_session_time_constants[bb_start,bb_end]
            param_ctr += 1;
            param_names += ["between block - " + str(block_types[bb_start]) + " to " + str(block_types[bb_end]) + " (units: sessions)"]
    
    # remove unused params
    vv = [np.any(~np.isinf(D[:,:,xx])) for xx in range(D.shape[2])];
    D = D[:,:,vv];
    param_names = np.array(param_names)[vv].tolist();
    
    if(vectorize_parameters):
        params = params[vv]
        return (D, param_names, params)
    else:
        return (D, param_names)


# TODO: Function to take a set of sessions and return a hddCRPModel
def create_hddCRP(seqs : list[ArrayLike], block_ids : ArrayLike, depth : int = 3, alpha_0 : float | ArrayLike = 1,
        weight_params_0 : float | ArrayLike = 1, weight_func : Callable = lambda x, y : np.exp(-np.sum(np.abs(x)/y)) ):

    Y = np.concatenate([np.array(ss).flatten() for ss in seqs], axis=0)
    groupings = create_context_tree(seqs, depth=depth)
    D, distance_labels = create_distance_matrix(seqs, block_ids)


    return hddCRPModel(Y, groupings, alpha_0, D,
                       weight_params=weight_params_0, weight_func=weight_func, weight_param_labels=distance_labels)
