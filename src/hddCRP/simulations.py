
import numpy as np
from numpy.typing import ArrayLike
import warnings

def simulate_sessions(session_lengths : ArrayLike, session_labels : ArrayLike,
                      num_responses : int, alpha : float, different_context_weights : ArrayLike,
                      within_session_timescales : dict, between_session_timescales : dict = None, repeat_bias_1_back : float = 1, base_measure : ArrayLike = None,
                      rng : np.random.Generator = None, repeat_bias_in_connection_weights : bool = False):
    if(rng is None):
        rng = np.random.default_rng()
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
    log_different_context_weights = np.log(1.0-different_context_weights)
    if(not (repeat_bias_1_back is None)):
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

    printed_within_timescale = {str(xx) : False for xx in within_session_timescales.keys()}
    printed_repeat_bias_1_back = False
    printed_different_context_weights = [False for xx in range(depth)]
    print("alpha = " + str(alpha))
    # raise RuntimeError()

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
        if(not (repeat_bias_1_back is None)):
            if(one_back >= 0):
                base_measure_c[one_back] *= repeat_bias_1_back
                base_measure_c /= np.sum(base_measure_c)
                if(not printed_repeat_bias_1_back):
                    print("repeat_bias_1_back = " + str(repeat_bias_1_back))
                    print("\tbase measure" + str(base_measure_c))
                    printed_repeat_bias_1_back= True
        # print(base_measure_c)

        # set up previous observations
        ws = np.zeros(tt)
        in_session = tts_session[:tt] == session
        tau_within = (within_session_timescales[labels[tt]]);
        dts = (trial_time - tts_trial[:tt][in_session]);
        ws[in_session] += -dts/tau_within
        if(not printed_within_timescale[str(labels[tt])]):
            printed_within_timescale[str(labels[tt])] = True
            print("tau " + str(labels[tt]) + " = " + str(tau_within) )

        #print(ws[in_session])

        if(not (between_session_timescales is None)):
            out_session = ~in_session
            dts = (session - tts_session[:tt][out_session])
            taus = np.array([between_session_timescales[xx, labels[tt]] for xx in labels[:tt][out_session]],dtype=float)
            ws[out_session] += -dts/taus

        context_match = contexts[:tt, :] == np.reshape(contexts[tt, :], (1,depth))
        for dd in range(depth):
            different_context = np.logical_not(np.all(context_match[:,:dd+1], axis=1))
            ws[different_context] += log_different_context_weights[dd]
            if(not printed_different_context_weights[dd]):
                print("different_context_weights " + str(dd) + " = " + str(different_context_weights[dd]))
                printed_different_context_weights[dd] = True

        if(repeat_bias_in_connection_weights and (not (repeat_bias_1_back is None))):
            ws[Y[:tt] == one_back] += log_repeat_bias_1_back

        # sum up observations
        ws_e = np.exp(ws)
        ts = np.zeros(num_responses)
        for ii in range(num_responses):
            ts[ii] += np.sum(ws_e[Y[:tt] == ii]) + alpha*base_measure_c[ii]

        
        ps = ts / np.sum(ts)

        Y[tt] = rng.choice(num_responses, p=ps);
        if(tt < 5):
            print(f"t {tt}: {ps}, {ts}, {Y[tt]}")

    session_starts = np.concatenate([[0], np.cumsum(session_lengths)])

    seqs = [Y[ss:ss+ll] for ss,ll in zip(session_starts, session_lengths)]
    return seqs

from itertools import product

def _hmm_ctrs(pattern : np.ndarray, depth : int, M : int, verbose : bool = False):
    pattern = pattern.ravel();

    if(depth > 0):
        cts = np.zeros((M,) * (depth + 1), dtype=int)
        if(verbose):
            print(f"pattern = {pattern}")

        for xx in product(range(M), repeat=depth):
            history = xx
            if(verbose):
                print(f"history = {history}")
            idx = np.ones(pattern.shape, dtype=bool)
            for ii, yy in enumerate(history):
                jj = ii + 1
                idx[:jj] = False
                idx[jj:] = idx[jj:] & (pattern[:-jj]==yy)
                if(verbose):
                    print(f"\tii = {ii}, yy = {yy}: {idx}")

            pattern_c = pattern[idx]
            hh = np.bincount(pattern_c, minlength=M)[:M];
            cts[np.s_[history]][:] = hh
            if(verbose):
                print(f"\t pattern_c = {pattern_c}, hh = {hh}")
                print(f"\t cts[np.s_[history]] = {cts[np.s_[history]]}")
                print(f"\t cts = {cts}")
    else:
        cts = np.bincount(pattern, minlength=M)[:M]

    return cts;



def construct_default_hmm(normalize : bool = True, alpha_pre : float = 0, alpha_post : float = 0):
    M = 3;
    H_pattern  = np.array([0,0,2,2,0,0])
    T2_pattern = np.array([0,1,1,2,0,1])

    P_pattern  = np.array([0,2,1,1,0,2])

    p_pre = (_hmm_ctrs(H_pattern[:4], 0, M) + _hmm_ctrs(T2_pattern[:4], 0, M))/2 
    t_pre = (_hmm_ctrs(H_pattern[:5], 1, M) + _hmm_ctrs(T2_pattern[:5], 1, M))/2
    q_pre = (_hmm_ctrs(H_pattern[:6], 2, M) + _hmm_ctrs(T2_pattern[:6], 2, M))/2

    p_post = _hmm_ctrs(P_pattern[:4], 0, M)
    t_post = _hmm_ctrs(P_pattern[:5], 1, M)
    q_post = _hmm_ctrs(P_pattern[:6], 2, M)


    for ii in range(M):
        if(np.all(t_pre[ii,:] == 0)):
            t_pre[ii,:] = p_pre
        if(np.all(t_post[ii,:] == 0)):
            t_post[ii,:] = p_post

    for ii in range(M):
        for jj in range(M):
            if(np.all(q_pre[ii,jj,:] == 0)):
                q_pre[ii,jj,:] = t_pre[ii,:]
            if(np.all(q_post[ii,jj,:] == 0)):
                q_post[ii,jj,:] = t_post[ii,:]

    p_pre = p_pre + alpha_pre
    t_pre = t_pre + alpha_pre
    q_pre = q_pre + alpha_pre
    p_post = p_post + alpha_post
    t_post = t_post + alpha_post
    q_post = q_post + alpha_post

    if(normalize):
        p_pre  = p_pre / np.sum(p_pre)
        p_post = p_post / np.sum(p_post)

        t_pre = t_pre / np.sum(t_pre,axis=-1,keepdims=True)
        t_post = t_post / np.sum(t_post,axis=-1,keepdims=True)

        q_pre = q_pre / np.sum(q_pre,axis=-1,keepdims=True)
        q_post = q_post / np.sum(q_post,axis=-1,keepdims=True)


    return p_pre, t_pre, q_pre, p_post, t_post, q_post

def upsample_set(p_s : list[np.ndarray]) -> list[np.ndarray]:

    for p_0 in p_s:
        for jj, p_1 in enumerate(p_s):
            while(p_1.ndim < p_0.ndim):
                p_1 = np.repeat(p_1[np.newaxis, ...], p_0.shape[0], axis=0)
                p_s[jj] = p_1
    return p_s

def interp_distributions(T_s : list[int] = [50],
                         p_s : list[np.ndarray] = [np.array([0.1, 0.1, 0.8]), np.array([0.4, 0.4, 0.2])]) -> np.ndarray:
    if(isinstance(T_s,int)):
        T_s = [T_s]

    p_s = upsample_set(p_s)
    for ii, T_c in enumerate(T_s):
        p = np.row_stack((p_s[ii].ravel(), p_s[ii+1].ravel()));

        if(ii+1 == len(T_s)):
            N = T_c-1
        else:
            N = T_c

        x  = np.arange(T_c)/N
        xp = np.arange(2)

        py = np.zeros((T_c, p.shape[1]))
        for yy in range(p.shape[1]):
            py[:,yy] = np.interp(x, xp, p[:,yy])

        Q_c = py.reshape((T_c,)+p_s[ii].shape);
        if(ii == 0):
            Q = Q_c;
        else:
            Q = np.concatenate((Q, Q_c), axis=0)
    return Q

def stack_distributions(T_s : list[int] = [50, 50],
                         p_s : list[np.ndarray] = [np.array([0.1, 0.1, 0.8]), np.array([0.4, 0.4, 0.2])]) -> np.ndarray:
    
    p_s = upsample_set(p_s)
    return np.vstack([np.repeat(p_c[np.newaxis,...], T_c, axis=0) for T_c, p_c in zip(T_s, p_s)])


def sim_markov(p_s : np.ndarray, p_0 : np.ndarray, rng : np.random.Generator = None) -> np.ndarray:
    if(rng is None):
        rng = np.random.default_rng()
    M = p_s.shape[-1]
    T = p_s.shape[0]

    D = p_s.ndim-2

    Y = np.zeros((T), dtype=int)
    Y[:D] = rng.choice(M, (D), p=p_0)

    for tt in range(D, T):
        p_c = p_s[tt,...]
        for dd in range(1,D+1):
            p_c = p_c[Y[tt-dd],...]
        Y[tt] = rng.choice(M, (1), p=p_c)
    return Y
