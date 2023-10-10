import numpy as np
from numpy.typing import ArrayLike
from itertools import product


from scipy.stats import gamma

from hddCRP.modelFittingSequential import sequentialhddCRPModel, DualAveragingForStepSize

EMPTY_INDICATOR = "NULL"
ALL_BLOCK_TYPES = "ALL_SESSIONS"


def distance_function_for_maze_task(log_params, D, B, num_layers,  num_timescales):
    timescales = np.exp(log_params[:num_timescales])
    log_scales = log_params[num_timescales:]
    log_scales = np.reshape(log_scales, (1,1,log_scales.size) + D.shape[3:])
    
    F = np.zeros(D.shape[:2] + (num_layers,) + D.shape[3:],dtype=float);

    distances_0 = np.abs(D[:,:,[0],...])
    distances = np.zeros_like(distances_0)
    distances.fill(-np.inf)
    timescale_inds = D[:,:,[1],...]
    for tt_ind, tt in enumerate(timescales):
        distances[timescale_inds == tt_ind] = -distances_0[timescale_inds == tt_ind]/tt

    if(D.shape[2] > 2 and len(log_scales) > 0):
        try:
            S = np.array([])
            S = D[:,:,num_layers:1:-1,...] * log_scales[::-1];
            F[:,:,:num_layers-1,...] = np.cumsum(S,axis=2)
        except:

            print("S " + str(S.shape))
            print("F " + str(F.shape))
            print("F[:,:,(num_layers-1):0:-1,...] " + str(F[:,:,(num_layers-1):0:-1,...].shape))
            print("D[:,:,2:,...] " + str(D[:,:,2:,...].shape))
            print("D " + str(D.shape))
            print("log_scales " + str(log_scales.shape))
            print("log_params " + str(log_params))
            raise RuntimeError("Error found")
        
    F += distances
    F = np.exp(F)
    
    if(len(log_scales) > 0):
        try:
            basemeasure_scale = np.exp(np.sum(B * log_scales, axis=2))
        except:

            print("log_scales " + str(log_scales.shape))
            print("B " + str(B.shape))
            raise RuntimeError("Error found")
    else:
        basemeasure_scale = None

    return F, basemeasure_scale

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
def create_distance_matrix(seqs : list[ArrayLike], block_ids : ArrayLike, actions = None,
                                distinct_within_session_distance_params : bool = True,
                                nback_scales : int = 2):
    block_ids = np.array(block_ids).flatten()
    assert len(seqs) == len(block_ids), "number of sequences and block_ids must be the same (number of trials)"

    # num_sessions = len(seqs)

    session_lengths       = [np.size(xx) for xx in seqs];
    session_start_indexes = np.append(0,np.cumsum(session_lengths));
    total_observations    = np.sum(session_lengths);

    block_types = np.unique(block_ids);
    B = len(block_types)

    ##
    D = np.zeros((total_observations, total_observations, 2 + nback_scales))
    D[:,:,0] = -np.inf
    D[:,:,1] = -1
    variable_ctr = 0;
    variable_names = [];

    if(actions is None):
        actions       = np.unique([np.unique(xx) for xx in seqs]);
    actions = np.array(actions)
    actions.sort()

    B = np.zeros((total_observations, len(actions), nback_scales))


    within_session_parameter_num  = np.zeros((len(B)), dtype=int)-1
    between_session_parameter_num = np.zeros((len(B), len(B)), dtype=int)-1

    for session_num_to in range(len(seqs)):
        
        to_block_idx = np.where(block_types == block_ids[session_num_to])[0]

        for session_num_from in range(session_num_to):
            from_block_idx = np.where(block_types == block_ids[session_num_from])[0]

            to_from_idx_0 = np.s_[session_start_indexes[session_num_to]:session_start_indexes[session_num_to+1], session_start_indexes[session_num_from]:session_start_indexes[session_num_from+1],0 ]
            to_from_idx_1 = np.s_[session_start_indexes[session_num_to]:session_start_indexes[session_num_to+1], session_start_indexes[session_num_from]:session_start_indexes[session_num_from+1],1 ]


            if(between_session_parameter_num[to_block_idx, from_block_idx] < 0):
                variable_names += [{"type" : "session_timeconstant", "from" : block_types[from_block_idx], "to" : block_types[to_block_idx]}]
                between_session_parameter_num[to_block_idx, from_block_idx] = variable_ctr
                variable_ctr += 1

            D[to_from_idx_0] = session_num_to-session_num_from
            D[to_from_idx_1] = between_session_parameter_num[to_block_idx, from_block_idx]

        # within session
        to_idx_0 = np.s_[session_start_indexes[session_num_to]:session_start_indexes[session_num_to+1], session_start_indexes[session_num_to]:session_start_indexes[session_num_to+1],0 ]
        to_idx_1 = np.s_[session_start_indexes[session_num_to]:session_start_indexes[session_num_to+1], session_start_indexes[session_num_to]:session_start_indexes[session_num_to+1],1 ]


        if(within_session_parameter_num[to_block_idx] < 0):
            if(distinct_within_session_distance_params):
                variable_names += [{"type" : "trial_timeconstant", "label" : block_types[to_block_idx]}]
                within_session_parameter_num[to_block_idx] = variable_ctr
                variable_ctr += 1
            else:
                variable_names += [{"type" : "trial_timeconstant", "label" : ALL_BLOCK_TYPES}]
                within_session_parameter_num[:] = variable_ctr
                variable_ctr += 1

        ds = np.arange(session_start_indexes[session_num_to], session_start_indexes[session_num_to+1], dtype=float)
        ds = ds[:,np.newaxis] - ds[np.newaxis,:];
        ds[ds <= 0] = -np.inf
        D[to_idx_0] = ds
        cs = np.ones_like(ds) * within_session_parameter_num[to_block_idx]
        cs[ds <= 0] = -1;
        D[to_idx_1] = cs

    num_timeconstants = variable_ctr


    Y = np.concatenate([np.array(ss).flatten() for ss in seqs])
    for nback in range(1,nback_scales+1):

        variable_names += [{"type" : "nback_scales", "label" :  nback}]
        variable_ctr += 1


        seqs_c = [np.roll(np.array(ss,dtype=object).flatten(),nback) for ss in seqs]
        for ss in seqs_c:
            ss[:nback] = np.nan 
        Y_prev = np.concatenate([np.array(ss).flatten() for ss in seqs_c])

        
        vv = Y_prev[:,np.newaxis] == Y[np.newaxis,:];
        vv = np.tril(vv,-1)
        D[:, :, 1 + nback] = vv

        for ii, action in enumerate(actions.astype(Y.dtype)):
            B[:,ii,nback-1] = (Y_prev == action)
            # print(Y_prev == action)
            # print(Y_prev )
            # print(Y )
            # print(action)
            # raise RuntimeError("Here!")

    return (D,  B, variable_names, num_timeconstants)

def parameter_vectorizer_for_distance_matrix(variable_names, session_types, within_session_time_constant = 50, between_session_time_constants = 10, nback_scales = 1):
    ## set up parameters in a specific vectorized form
    # param_names = ["" for xx in range(len(within_session_vars) + len(between_session_vars)*2)];

    params_vector = np.zeros((len(variable_names)))
    is_within_timescale = np.zeros((len(variable_names)), dtype=bool)
    is_between_timescale = np.zeros((len(variable_names)), dtype=bool)
    is_nback_scale = np.zeros((len(variable_names)), dtype=bool)

    unique_session_types = np.unique(session_types);

    param_names = []


    for var_idx, var in enumerate(variable_names):
        if(var["type"] == "trial_timeconstant"):
            if(var["label"] == ALL_BLOCK_TYPES):
                pos = 0;
                param_names += ["within_session_time_constant"]
            else:
                pos = np.where(unique_session_types == var["label"])[0][0];
                param_names += ["within_session_" + str(var["label"]) + "_time_constant"]

            if(np.isscalar(within_session_time_constant)):
                params_vector[var_idx] = within_session_time_constant
            else:
                params_vector[var_idx] = within_session_time_constant[pos]
            is_within_timescale[var_idx] = True
        elif(var["type"] == "session_timeconstant"):
            pos_from = np.where(unique_session_types == var["from"])[0][0];
            pos_to = np.where(unique_session_types == var["to"])[0][0];
            
            
            param_names += [str(var["from"]) + "_to_" + str(var["to"]) + "_session_time_constant"];

            if(np.isscalar(between_session_time_constants)):
                params_vector[var_idx] = between_session_time_constants
            else:
                params_vector[var_idx] = between_session_time_constants[pos_from,pos_to]
            is_between_timescale[var_idx] = True
        elif(var["type"] == "nback_scales"):

            param_names += ["scale_" + str(var["label"]) + "_back"]

            if(np.isscalar(nback_scales)):
                params_vector[var_idx] = nback_scales
            else:
                params_vector[var_idx] = nback_scales[var["label"]-1]
            is_nback_scale[var_idx] = True
        else:
            raise ValueError("Unknown paramter type")
    
    return (param_names, params_vector, is_within_timescale, is_between_timescale, is_nback_scale)

# TODO: Function to take a set of sessions and return a hddCRPModel
def create_hddCRP(seqs : list[ArrayLike], block_ids : ArrayLike, depth : int = 3, alpha_0 : float | ArrayLike = None, actions = None,
        weight_params_0 : float | ArrayLike = None, rng : np.random.Generator = None, fit_nback_scales : bool = True):
    Y = np.concatenate([np.array(ss).flatten() for ss in seqs], axis=0)
    if(actions == None):
        actions = np.unique(Y)
    actions = np.array(actions)
    actions.sort()

    block_ends   = np.cumsum(np.array([np.size(ss) for ss in seqs],dtype=int))-1
    block_starts = np.cumsum(np.array([0] + [np.size(ss) for ss in seqs],dtype=int))
    groupings = create_context_tree(seqs, depth=depth)

    D, B, distance_labels, num_timescales = create_distance_matrix(seqs, block_ids,
                                                distinct_within_session_distance_params = True,
                                                nback_scales = fit_nback_scales*(depth-1), actions=actions);

    # set up distances for predictive distributions
    prefixes = [str(xx) + '-' for xx in actions]
    combinations = list(product(prefixes,repeat=depth-1))

    prediction_groups_at_each_level = [[]] * depth
    prediction_groups_at_each_level[0] = ['' for xx in combinations]
    for ii in range(1,depth):
        prediction_groups_at_each_level[ii] = [''.join(xx[-ii:]) for xx in combinations]
    contexts = [''.join(xx) for xx in combinations]
    observation_indices = block_ends;

    num_contexts = len(contexts)

    D_pred = np.zeros((num_contexts, D.shape[1], D.shape[2], len(seqs)))
    B_pred = np.zeros((num_contexts, B.shape[1], B.shape[2], len(seqs)))

    # get weights for specified observations
    for ii in range(len(seqs)):
        seq_idx = range(block_starts[ii],block_starts[ii+1])
        for jj, con in enumerate(contexts):
            seqs_c = [ss.copy() for ss in seqs[:(ii+1)]]
            seqs_c[-1] = np.concatenate([np.array(seqs_c[-1]), np.array(list(con)), np.array([np.nan])])

            D_c, B_c, *_ = create_distance_matrix(seqs_c, block_ids[:(ii+1)],
                                            distinct_within_session_distance_params = True,
                                            nback_scales = depth-1, actions=actions);
            D_c = D_c[-1,:block_starts[ii+1], :]

            D_c[block_starts[ii]:block_starts[ii+1],0] -= len(con)
            D_pred[jj, :block_starts[ii+1], :, ii] = D_c

            B_pred[jj, :, :, ii] = B_c[-1,:,:]
    

    if(alpha_0 is None):
        alpha_0 = rng.gamma(shape=2, scale=10, size=(depth))

    param_names, params_vector, is_within_timescale, is_between_timescale, is_nback_scale = parameter_vectorizer_for_distance_matrix(distance_labels, np.unique(block_ids))
    param_types = {"is_within_timescale" : is_within_timescale, "is_between_timescale" : is_between_timescale, "is_nback_scale" : is_nback_scale}
    

    if(not weight_params_0 is None):
        params_vector[:] = weight_params_0
    else:
        params_vector[is_within_timescale]  = rng.gamma(shape=2,  scale=20, size=(np.sum(is_within_timescale)))
        params_vector[is_between_timescale] = rng.gamma(shape=2,  scale=5, size=(np.sum(is_between_timescale)))
        params_vector[is_nback_scale]       = rng.gamma(shape=20, scale=1/20, size=(np.sum(is_nback_scale)))

    weight_func =  lambda D, B, log_params, num_layers : distance_function_for_maze_task(log_params, D, B, num_layers,  num_timescales)

    model = sequentialhddCRPModel(Y, groupings, alpha_0, D=D, B=B,
                       weight_params=np.log(params_vector),  weight_param_labels=param_names, weight_func=weight_func, rng=rng)
    model._param_types = param_types
    model._block_ends  = block_ends


    model.setup_transition_probability_computations(prediction_groups_at_each_level, observation_indices, contexts, D_pred, B_pred)
    return model



def compute_kl_diveregences_between_transition_probabilities(probs_1, probs_2, contexts_1, contexts_2,rng=None):
    '''
    Args:
        probs_1: (samples x M x len(contexts_1) x number of points)
        probs_2: (samples x M x len(contexts_2) x number of points)

    Returns
        (divergences, contexts)

        divegences: (array) samples X [KL(deep context || shallow context), KL(shallow context || deep context), JS(shallow context, deep context)] x len(contexts) x number of points
        contexts: list of tuples of string pairs for the contexts. (shallow context, deep context)
    '''
    if(probs_1.ndim == 3):
        probs_1 = probs_1[:,:,:,np.newaxis]
    if(probs_2.ndim == 3):
        probs_2 = probs_2[:,:,:,np.newaxis]
    assert probs_1.ndim == 4, "probs_1 should have 4 dimensions"
    assert probs_2.ndim == 4, "probs_2 should have 4 dimensions"
    # assert probs_1.shape[0] == probs_2.shape[0], "probs_1.shape[0] must equal probs_2.shape[0] (this requirement could be loosened)"
    assert probs_1.shape[1] == probs_2.shape[1], "probs_1.shape[1] must equal probs_2.shape[1]"
    assert probs_1.shape[3] == probs_2.shape[3], "probs_1.shape[3] must equal probs_2.shape[3]"
    assert probs_1.shape[2] == len(contexts_1), "probs_1.shape[2] doesn't match len(context_1)"
    assert probs_2.shape[2] == len(contexts_2), "probs_2.shape[2] doesn't match len(context_2)"

    if(rng is None):
        rng = np.random.default_rng();
    if(probs_1.shape[0] < probs_2.shape[0]):
        probs_1 = probs_1[rng.integers(0, probs_1.shape[0],probs_2.shape[0]), :, : ,:]
    elif(probs_1.shape[0] > probs_2.shape[0]):
        probs_2 = probs_2[rng.integers(0, probs_2.shape[0],probs_1.shape[0]), :, : ,:]



    contexts_1 = [[x for x in cc.split('-') if x != ''] for cc in contexts_1]
    contexts_2 = [[x for x in cc.split('-') if x != ''] for cc in contexts_2]


    assert np.all([len(xx) == len(contexts_1[0]) for xx in contexts_1]), "contexts_1 must all be of same depth"
    assert np.all([len(xx) == len(contexts_2[0]) for xx in contexts_2]), "contexts_2 must all be of same depth"

    # match up the contexts
    if(len(contexts_1[0]) < len(contexts_2[0])):
        contexts_short = contexts_1;
        contexts_long = contexts_2;

        probs_short = probs_1;
        probs_long = probs_2;
    else:
        contexts_short = contexts_2;
        contexts_long = contexts_1;

        probs_short = probs_2;
        probs_long = probs_1;
    
    num_contexts = len(contexts_long)
    short_context_indices = np.zeros((num_contexts),dtype=int)
    for ii_i, ii in enumerate(contexts_long):
        match_index = -1;
        for jj_i, jj in enumerate(contexts_short):
            if((len(jj) == 0) or (jj == ii[-len(jj):])):
                match_index = jj_i
                break
        if(match_index < 0):
            raise RuntimeError("No matching contexts found")
        short_context_indices[ii_i] = match_index;

    num_obs = probs_1.shape[3]
    context_strs = []
    divs = np.zeros((probs_short.shape[0], 3, num_contexts, num_obs))
    for jj in range(num_obs):
        for ii in range(num_contexts):
            ii_short = short_context_indices[ii];

            P = probs_long[:,:,ii,jj];
            Q = probs_short[:,:,ii_short,jj];
            M = 0.5*(P + Q)

            lP = np.log(P)
            lQ = np.log(Q)
            lM = np.log(M)

            divs[:, 0, ii, jj] = np.sum(P * (lP - lQ), axis=1)
            divs[:, 1, ii, jj] = np.sum(Q * (lQ - lP), axis=1)
            divs[:, 2, ii, jj] = 0.5*(np.sum(P * (lP - lM), axis=1) + np.sum(Q * (lQ - lM), axis=1))

            context_strs.append(('-'.join(contexts_short[ii_short]) + "-", '-'.join(contexts_long[ii]) + "-"))

    return divs, context_strs, short_context_indices

    

def Metropolis_Hastings_step_for_maze_data(hddcrp : sequentialhddCRPModel | list[sequentialhddCRPModel], sigma2 : ArrayLike | float, uniform_prior : bool = False, 
        prior_shapes = None, prior_scales=None, single_concentration_parameter=False) -> tuple[sequentialhddCRPModel, float]:
    '''
    Takes a random-walk Metropolis-Hastings step for the hddCRP model parameters.
    Here, I assume all parameters can take any float value - the prior must do the transform for any constraints.

    Args:
        hddcrp: The model to sample from.
        sigma2: (scalar, vector of length hddcrp.num_parameters, or matrix size hddcrp.num_parameters x hddcrp.num_parameters) The size of the MH proposal step in each dimension.
               If scalar, covariance of step is EYE(num_parameters)*sigma
               If vector, covariance of step diag(sigma)
               If matrix, is the full covariance matrix
    Returns:
        (hddcrp, log_acceptance_probability, accepted)
        hddcrp: The hddCRPModel object.
        log_acceptance_probability: (float) the log acceptance probability in the MH step
        accepted: (bool) whether or not the sample was accepted (hddcrp is changed if True)
    Raises:
        ValueError: if sigma2 is incorrect shape
    '''
    if isinstance(hddcrp, list):
        multi_model = True;
        hddcrp_0 = hddcrp[0]
    else:
        hddcrp_0 = hddcrp
        multi_model = False;

    if(single_concentration_parameter):
        num_alphas = 1;
        theta_current = np.concatenate([[np.log(hddcrp_0.alpha[0])], hddcrp_0.weight_params.flatten()]);
    else:
        num_alphas = hddcrp_0.alpha.size;
        theta_current = np.concatenate([np.log(hddcrp_0.alpha), hddcrp_0.weight_params.flatten()]);
    weight_idx = range(num_alphas, hddcrp_0.weight_params.size + num_alphas)
    alpha_idx = range(num_alphas)

    if(np.isscalar(sigma2)):
        eps = hddcrp_0._rng.normal(scale=np.sqrt(sigma2),size=theta_current.shape)
    elif(np.size(sigma2) == np.size(theta_current)):
        eps = hddcrp_0._rng.normal(scale=np.sqrt(np.array(sigma2).flatten()))
    elif(np.shape(sigma2) == (np.size(theta_current),)*2):
        eps = hddcrp_0._rng.multivariate_normal(np.zeros_like(theta_current), sigma2)
    else:
        raise ValueError("invalid variance for random-walk Metropolis-Hastings step: " + str(sigma2))

    theta_star = theta_current + eps;

    log_alpha_curr = theta_current[alpha_idx]
    log_alpha_star = theta_star[alpha_idx]
    weight_curr = theta_current[weight_idx]
    weight_star = theta_star[weight_idx]

    if(multi_model):
        log_p_Y_current = np.array([hddcrp_c.compute_log_likelihood() for hddcrp_c in hddcrp])
        log_p_Y_star    = np.array([hddcrp_c.compute_log_likelihood(alphas=np.exp(log_alpha_star), weight_params=theta_star[weight_idx]) for hddcrp_c in hddcrp])
    else:
        log_p_Y_current = hddcrp.compute_log_likelihood()
        log_p_Y_star    = hddcrp.compute_log_likelihood(alphas=np.exp(log_alpha_star), weight_params=theta_star[weight_idx])

    w_idx = hddcrp_0._param_types["is_within_timescale"]
    b_idx = hddcrp_0._param_types["is_between_timescale"]
    s_idx = hddcrp_0._param_types["is_nback_scale"]
    

    if(uniform_prior):
        raise NotImplementedError("Should not be running in this mode for production")
    else:
        if(prior_shapes is None):
            prior_shapes = {"alpha" : 2,
                            "tau_within" : 2,
                            "tau_between" : 2,
                            "nback" : 20}
        if( prior_scales is None):
            prior_scales = {"alpha" : 5,
                            "tau_within" : 25,
                            "tau_between" : 5,
                            "nback" : 1/20}
        
        
        log_P_theta_current = 0;
        log_P_theta_current += np.sum(gamma.logpdf(np.exp(log_alpha_curr),     prior_shapes["alpha"], scale=prior_scales["alpha"])             + log_alpha_curr)
        log_P_theta_current += np.sum(gamma.logpdf(np.exp(weight_curr[w_idx]), prior_shapes["tau_within"], scale=prior_scales["tau_within"])   + weight_curr[w_idx])
        log_P_theta_current += np.sum(gamma.logpdf(np.exp(weight_curr[b_idx]), prior_shapes["tau_between"], scale=prior_scales["tau_between"]) + weight_curr[b_idx])
        log_P_theta_current += np.sum(gamma.logpdf(np.exp(weight_curr[s_idx]), prior_shapes["nback"], scale=prior_scales["nback"])             + weight_curr[s_idx])

        log_P_theta_star = 0;
        log_P_theta_star += np.sum(gamma.logpdf(np.exp(log_alpha_star),     prior_shapes["alpha"], scale=prior_scales["alpha"])             + log_alpha_star)
        log_P_theta_star += np.sum(gamma.logpdf(np.exp(weight_star[w_idx]), prior_shapes["tau_within"], scale=prior_scales["tau_within"])   + weight_star[w_idx])
        log_P_theta_star += np.sum(gamma.logpdf(np.exp(weight_star[b_idx]), prior_shapes["tau_between"], scale=prior_scales["tau_between"]) + weight_star[b_idx])
        log_P_theta_star += np.sum(gamma.logpdf(np.exp(weight_star[s_idx]), prior_shapes["nback"], scale=prior_scales["nback"])             + weight_star[s_idx])

    log_acceptance_probability = min(0.0, np.sum(log_p_Y_star) + log_P_theta_star - (np.sum(log_p_Y_current) + log_P_theta_current))

    like_diff = np.sum(log_p_Y_star) - np.sum(log_p_Y_current)
    prior_diff = log_P_theta_star - log_P_theta_current

    aa = -np.random.exponential()

    if(aa < log_acceptance_probability):
        accepted = True
        if(multi_model):
            for hddcrp_c in hddcrp:
                hddcrp_c.alpha = np.exp(log_alpha_star)
                hddcrp_c.weight_params = weight_star
        else:
            hddcrp.alpha = np.exp(log_alpha_star)
            hddcrp.weight_params = weight_star
        log_like = log_p_Y_star
        log_prior = log_P_theta_star
    else:
        accepted = False
        log_like = log_p_Y_current
        log_prior = log_P_theta_current

    return (hddcrp, log_acceptance_probability, accepted, log_like, log_prior, like_diff, prior_diff)


def sample_model_for_maze_data(hddcrp : sequentialhddCRPModel, num_samples : int, num_warmup_samples : int,
            uniform_prior : bool = False, print_every : int = None, prior_shapes = None, prior_scales=None, single_concentration_parameter=False,
            compute_transition_probabilties=True):
    num_samples = int(num_samples)
    num_warmup_samples = int(num_warmup_samples)
    assert num_samples > 0, "must sample positive number of values"
    assert num_warmup_samples >= 0, "must sample positive number of values"

    num_samples_total = num_samples + num_warmup_samples

    step_size_settings = DualAveragingForStepSize();

    if(single_concentration_parameter):
        hddcrp.alpha = hddcrp.alpha[0];

    if(compute_transition_probabilties):
        contexts = hddcrp._predictive_transition_probability_setup["contexts"]
        observation_indices = hddcrp._predictive_transition_probability_setup["observation_indices"]

        transition_probabilities = { "contexts" : contexts,
            "observation_indices" : observation_indices,
            "probabilities" : np.zeros((num_samples_total, hddcrp.M, len(contexts), len(observation_indices))),
            "sampled_probabilities" : np.zeros((num_samples_total, hddcrp.M, len(contexts), len(observation_indices)))
        }
    
    samples = {"log_acceptance_probability" : np.zeros((num_samples_total)),
               "log_like_diff" : np.zeros((num_samples_total)),
               "log_prior_diff" : np.zeros((num_samples_total)),
               "log_like" : np.zeros((num_samples_total)),
               "log_prior" : np.zeros((num_samples_total)),
               "accepted" : np.zeros((num_samples_total),dtype=bool),
               "alphas"   : np.zeros((num_samples_total,hddcrp.alpha.size)),
               "log_taus" : np.zeros((num_samples_total,hddcrp.weight_params.size)),
               "num_warmup_samples" : num_warmup_samples}

    
    for ss in range(num_samples_total):
        if((not print_every is None) and (ss % print_every == 0)):
            print("Sample " + str(ss) + " / " + str(num_samples_total))
        sigma2 = step_size_settings.step_size_fixed if ss >= num_warmup_samples else step_size_settings.step_size_for_warmup
        hddcrp, samples["log_acceptance_probability"][ss], samples["accepted"][ss], samples["log_like"][ss], samples["log_prior"][ss] , samples["log_like_diff"][ss], samples["log_prior_diff"][ss] = Metropolis_Hastings_step_for_maze_data(hddcrp, sigma2, uniform_prior=uniform_prior, prior_shapes = prior_shapes, prior_scales=prior_scales, single_concentration_parameter=single_concentration_parameter)

        samples["alphas"][ss,:] = hddcrp.alpha
        samples["log_taus"][ss,:] = hddcrp.weight_params

        if(compute_transition_probabilties):
            transition_probabilities["probabilities"][ss,:,:,:] = hddcrp.compute_preditive_transition_probabilities()
            transition_probabilities["sampled_probabilities"][ss,:,:,:] = hddcrp.sample_preditive_transition_probabilities()

        if(ss < num_warmup_samples):
            step_size_settings.update(np.exp(samples["log_acceptance_probability"][ss]))

    if(compute_transition_probabilties):
        samples["transition_probabilities"] = transition_probabilities;
    return (hddcrp, samples, step_size_settings)



def sample_population_model_for_maze_data(hddcrps : list[sequentialhddCRPModel], num_samples : int, num_warmup_samples : int,
            uniform_prior : bool = False, print_every : int = None, prior_shapes = None, prior_scales=None, single_concentration_parameter=False,
            prior_concentration_for_mixing : float = 1, num_components : int = 1, compute_transition_probabilties : bool = True):
    num_samples = int(num_samples)
    num_warmup_samples = int(num_warmup_samples)
    assert num_samples > 0, "must sample positive number of values"
    assert num_warmup_samples >= 0, "must sample positive number of values"

    if(num_components != 1):
        raise NotImplementedError("Not fully implemented for multi-component model")

    num_samples_total = num_samples + num_warmup_samples

    step_size_settings = DualAveragingForStepSize();

    num_models = len(hddcrps)

    for hddcrp in hddcrps:
        hddcrp.weight_params = hddcrps[0].weight_params;
        if(single_concentration_parameter):
            hddcrp.alpha = hddcrps[0].alpha[0];
        else:
            hddcrp.alpha = hddcrps[0].alpha;

    if(compute_transition_probabilties):
        transition_probabilities = []
        for hddcrp in hddcrps:
            contexts = hddcrp._predictive_transition_probability_setup["contexts"]
            observation_indices = hddcrp._predictive_transition_probability_setup["observation_indices"]

            transition_probabilities_c = { "contexts" : contexts,
                "observation_indices" : observation_indices,
                "probabilities" : np.zeros((num_samples_total, hddcrp.M, len(contexts), len(observation_indices))),
                "sampled_probabilities" : np.zeros((num_samples_total, hddcrp.M, len(contexts), len(observation_indices)))
            } 
            transition_probabilities += [transition_probabilities_c]

    
    samples = {"log_acceptance_probability" : np.zeros((num_samples_total)),
               "log_like_diff" : np.zeros((num_samples_total)),
               "log_prior_diff" : np.zeros((num_samples_total)),
               "log_like" : np.zeros((num_samples_total, num_models)),
               "log_prior" : np.zeros((num_samples_total, num_components)),
               "log_prior_mixing" : np.zeros((num_samples_total)),
               "log_p_state" : np.zeros((num_samples_total, num_models)),
               "accepted" : np.zeros((num_samples_total),dtype=bool),
               "state" : np.zeros((num_samples_total, num_models),dtype=int),
               "alphas"   : np.zeros((num_samples_total,hddcrps[0].alpha.size)), #,num_components
               "log_taus" : np.zeros((num_samples_total,hddcrps[0].weight_params.size)), #,num_components
               "num_warmup_samples" : num_warmup_samples}

    
    for ss in range(num_samples_total):
        if((not print_every is None) and (ss % print_every == 0)):
            print("Sample " + str(ss) + " / " + str(num_samples_total))
        sigma2 = step_size_settings.step_size_fixed if ss >= num_warmup_samples else step_size_settings.step_size_for_warmup
        hddcrps, samples["log_acceptance_probability"][ss], samples["accepted"][ss], samples["log_like"][ss,:], samples["log_prior"][ss,:] , samples["log_like_diff"][ss], samples["log_prior_diff"][ss] = Metropolis_Hastings_step_for_maze_data(hddcrps, sigma2, uniform_prior=uniform_prior, prior_shapes = prior_shapes, prior_scales=prior_scales, single_concentration_parameter=single_concentration_parameter)

        samples["alphas"][ss,:] = hddcrps[0].alpha
        samples["log_taus"][ss,:] = hddcrps[0].weight_params

        if(compute_transition_probabilties):
            for mm, hddcrp in enumerate(hddcrps):
                transition_probabilities[mm]["probabilities"][ss,:,:,:] = hddcrp.compute_preditive_transition_probabilities()
                transition_probabilities[mm]["sampled_probabilities"][ss,:,:,:] = hddcrp.sample_preditive_transition_probabilities()

        if(ss < num_warmup_samples):
            step_size_settings.update(np.exp(samples["log_acceptance_probability"][ss]))

    if(compute_transition_probabilties):
        samples["transition_probabilities"] = transition_probabilities;

    return (hddcrps, samples, step_size_settings)