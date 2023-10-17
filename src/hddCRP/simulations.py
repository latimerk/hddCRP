
import numpy as np
from numpy.typing import ArrayLike
import warnings

def simulate_sessions(session_lengths : ArrayLike, session_labels : ArrayLike,
                      num_responses : int, alpha : float, different_context_weights : ArrayLike,
                      within_session_timescales : dict, between_session_timescales : dict = None, repeat_bias_1_back : float = 1, base_measure : ArrayLike = None,
                      rng : np.random.Generator = None, repeat_bias_in_connection_weights : bool = False):
    if(np.isscalar(session_lengths)):
        session_lengths = [session_lengths];
    if(np.isscalar(session_labels)):
        session_lengths = [session_labels];
    
    assert len(session_labels) == len(session_lengths), "length of session_labels and session_lengths must match"
    session_labels = np.array(session_labels).flatten()
    session_lengths = np.array(session_lengths,dtype=int).flatten()
    assert np.all(session_lengths > 0), "all elements of session_lengths must be positive integers"

    different_context_weights = np.array(different_context_weights).flatten()
    depth = different_context_weights.size
    assert np.all(different_context_weights >= 0), "context_weights must be positive"
    if(np.any(different_context_weights > 1)):
        warnings.warn("Context weights in model are assumed to be <= 1. Values > 1 found.")
    log_different_context_weights = np.log(different_context_weights)
    if(repeat_bias_1_back is None):
        repeat_bias_1_back = 1
    assert np.isscalar(repeat_bias_1_back) and repeat_bias_1_back > 0, "repeat_bias_1_back must be a positive scalar"
    log_repeat_bias_1_back = np.log(repeat_bias_1_back)
    

    num_responses = int(num_responses)
    assert num_responses > 1, "must have at least 2 responses"
    
    assert alpha > 0, "alpha must be positive"

    if((base_measure is None) or (np.isscalar(base_measure))):
        base_measure = np.ones((num_responses))/num_responses;
    else:
        base_measure = np.array(base_measure).flatten()
        base_measure = base_measure/np.sum(base_measure)
    assert base_measure.size == num_responses, "base_measure must be of length num_responses"
    assert np.all(base_measure > 0), "base_measure must be all positive"


    T_total = np.sum(session_lengths)

    Y = np.zeros((T_total),dtype=int)
    contexts = np.zeros((T_total, depth),dtype=int)
    labels = np.concatenate([np.array([label] * length) for label, length in zip(session_labels,session_lengths)])
    tts_trial   = np.concatenate([np.arange(0, length) for length in session_lengths])
    tts_session = np.concatenate([np.ones(length) * session_num for session_num, length in enumerate(session_lengths)])

    if(rng is None):
        rng = np.random.default_rng()


    for tt in range(T_total):
        #
        session    = tts_session[tt];
        trial_time = tts_trial[tt]

        # build context
        one_back = -1
        
        tt_prev = tt-1
        if(tt_prev >= 0 and tts_session[tt_prev] == session):
            one_back = Y[tt_prev]

        contexts[tt,:] = -1
        for dd in range(depth):
            tt_prev = tt-(dd+1)
            if(tt_prev >= 0 and tts_session[tt_prev] == session):
                contexts[tt,dd] = Y[tt_prev]

        # get base measure
        base_measure_c = base_measure.copy()
        if(one_back >= 0):
            base_measure_c[one_back] *= repeat_bias_1_back
            base_measure_c /= np.sum(base_measure_c)
        # print(base_measure_c)

        # set up previous observations
        ws = np.zeros(tt)
        in_session = tts_session[:tt] == session
        tau_within = within_session_timescales[labels[tt]];
        dts = (trial_time - tts_trial[:tt][in_session]);
        ws[in_session] += -dts/tau_within

        if(not (between_session_timescales is None)):
            out_session = ~in_session
            ws[out_session] += -(trial_time - tts_session[:tt][out_session])/np.array([between_session_timescales[xx, labels[tt]] for xx in labels[:tt][out_session]])

        context_match = contexts[:tt, :] == np.reshape(contexts[tt, :], (1,depth))
        for dd in range(depth):
            different_context = ~np.all(context_match[:,:depth+1], axis=1)
            ws[different_context] += log_different_context_weights[dd]

        if(repeat_bias_in_connection_weights):
            ws[Y[:tt] == one_back] += log_repeat_bias_1_back

        # sum up observations
        ws_e = np.exp(ws)
        ts = np.zeros(num_responses)
        for ii in range(num_responses):
            ts[ii] += np.sum(ws_e[Y[:tt] == ii]) + alpha*base_measure_c[ii]

        
        # for ii in range(num_responses):
        #     ts[ii] = alpha/num_responses + sum(Y[:tt] == ii);

        ps = ts / np.sum(ts)

        Y[tt] = rng.choice(num_responses, p=ps);

    session_starts = np.concatenate([[0], np.cumsum(session_lengths)])

    seqs = [Y[ss:ss+ll] for ss,ll in zip(session_starts, session_lengths)]
    # print("tau_within")
    # print(tau_within)
    # print("ws")
    # print(ws)
    # print("ws_e")
    # print(ws_e)
    # print("ts")
    # print(ts)
    # print("ps")
    # print(ps)
    return seqs