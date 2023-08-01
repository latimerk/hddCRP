import numpy as np
from numpy.typing import ArrayLike
from typing import Callable

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

# TODO: Function to get distances from a set of time series actions (set of sessions)
def create_distance_matrix(seqs : list[ArrayLike], block_ids : ArrayLike,   distinct_within_session_distance_params : bool = False,
                                                                            sequential_within_session_distances : bool = False,
                                                                            sequential_between_session_same_block_distances : bool = False,
                                                                            sequential_between_session_different_block_distances : bool = True,
                                                                            within_block_disance_in_total_sessions : bool = True):
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

    ## Within session effects: should be sequential or all?
    for bb in range(B):
        for ss in session_block_indexes[bb]:
            rr = range(session_start_indexes[ss], session_start_indexes[ss+1])
            L = np.ones((len(rr), len(rr)));
            if(sequential_within_session_distances):
                L = np.tril(L);

            D[rr,rr,param_ctr] = L
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
                distance_in_same_block = end_num - start_num; # units: sessions
                distance_all_sessions = ss_end - ss_start; # units: sessions
                if(ss_start < ss_end or not sequential_between_session_same_block_distances):
                    rr_start = range(session_start_indexes[ss_start], session_start_indexes[ss_start+1])
                    rr_end   = range(session_start_indexes[ss_end  ], session_start_indexes[ss_end  +1])

                    D[rr_start, rr_end, param_ctr] = distance_all_sessions if within_block_disance_in_total_sessions else distance_in_same_block;
        param_ctr += 1;
        param_names += ["within block - " + str(block_types[bb]) + " (units: sessions)"]


    for bb_start in range(B):
        for bb_end in range(B):
            if(bb_start == bb_end):
                continue;

            for ss_start in session_block_indexes[bb_start]:
                for ss_end in session_block_indexes[bb_end):
                    if(ss_start < ss_end or not sequential_between_session_different_block_distances):
                        rr_start = range(session_start_indexes[ss_start], session_start_indexes[ss_start+1])
                        rr_end   = range(session_start_indexes[ss_end  ], session_start_indexes[ss_end  +1])
                        D[rr_start, rr_end, param_ctr] = ss_end - ss_start;
            param_ctr += 1;
            param_names += ["between block - " + str(block_types[bb_start]) + " to " + str(block_types[bb_end]) + " (units: sessions)"]
    
    # remove unused params
    vv = [np.any(~np.isinf(D[:,:,xx])) for xx in range(D.shape[2])];
    D = D[:,:,vv];
    param_names = np.array(param_names)[vv].tolist();
    
    return (D, param_names)


# TODO: Function to take a set of sessions and return a hddCRPModel
def create_hddCRP(seqs : list[ArrayLike], block_ids : ArrayLike, depth : int = 3, alpha_0 : float | ArrayLike = 1,
        weight_params_0 : float | ArrayLike = 1, weight_func : Callable = lambda x, y : np.exp(-np.sum(np.abs(x)/y)) ):

    Y = np.concatenate([np.array(ss).flatten() for ss in seqs], axis=0)
    groupings = np.concatenate([create_context_tree_single_session(ss, depth=depth) for ss in seqs], axis=0)
    D, distance_labels = create_distance_matrix(seqs, block_ids)


    return hddCRPModel(Y, groupings, alpha_0, D,
                       weight_params=weight_params_0, weight_func=weight_func, weight_param_labels=distance_labels)