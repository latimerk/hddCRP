from __future__ import annotations
import numpy as np;
import numpy.typing as npt

def get_transition_count(seq : npt.ArrayLike, depth : int = 2, conditions : npt.ArrayLike = None, symbols : int | list | None = None, return_symbols : bool = False) -> npt.NDArray[np.int_] | tuple[npt.NDArray[np.int_],npt.NDArray[np.int_]]:
    """ Counts all transitions observed in sequence to a specific depth.

    Args:
      seq: A 1-D list of observed symbols/actions.
      depth: The dependence on past observations. If 2, then counts Num(observations) given last two observations.
      symbols: The number of observation symbols, or a list of symbols. If None, is set to unique(seq)

    Returns:
      transition_count_matrix: Array of depth+1 dimensions. Each dimension is of size num_symbols.
        transition_count_matrix[C,B,A] = number of times the sequence (C,B,A) was observed

      If return_symbols:
      symbols: list of symbols in the order they appear in the transition_count_matrix. If symbols[1] = 'A', symbols[0] = 'B', symbols[2] = 'C'
               then transition_count_matrix[2,0,1] = transition_count_matrix['A','B','C']
    """
    seq = np.array(seq).ravel()

    if(isinstance(symbols, int)):
        symbols = np.arange(symbols,dtype=int)
    elif(symbols is None):
        symbols = np.unique(seq)
    else:
        symbols = np.unique(symbols)
    num_symbols = len(symbols)

    if(np.any(np.isin(seq, symbols, invert=True))):
        raise ValueError('seq contains values not in symbols')
    
    seq_0 = np.zeros(seq.size, dtype=int)
    for ii,ss in enumerate(symbols):
        seq_0[seq == ss] = ii
    
    if(conditions is not None):
        unique_conds, cond_idx = np.unique(conditions,return_inverse=True)
        num_conds = len(unique_conds)
        transition_count_matrix = np.zeros((num_conds,) + (num_symbols,)*(depth + 1))
    else:
        transition_count_matrix = np.zeros((num_symbols,)*(depth + 1))

    for ii in range(depth,len(seq_0)):
        if(conditions is not None):
            sub_seq = (cond_idx[ii-1],) + tuple(seq_0[ii-depth:(ii+1)])
        else:
            sub_seq = tuple(seq_0[ii-depth:(ii+1)])
        transition_count_matrix[sub_seq] += 1

    if(return_symbols):
        return transition_count_matrix, symbols
    else:
        return transition_count_matrix
    

def get_transition_probabilities(transition_count_matrix : npt.NDArray[np.int_], alpha : float | np.ndarray = 0, rng : np.random.Generator = None) -> npt.NDArray[np.float_]:
    """ Turns transitions counts into transition probabilities.
        Estimates can be regularized via an alpha term that acts as a Dirichlet prior.
        Samples from the posterior given the prior are drawn if a random number generator is provides.

    Args:
      transition_count_matrix: Array of depth+1 dimensions. Each dimension is of size num_symbols.
        transition_count_matrix[C,B,A] = number of times the sequence (C,B,A) was observed
      alpha: parameter of dirichlet prior. Set to 0 or None if not used.
             alpha can be a constant or an array of shape transition_count_matrix.shape[aa:] for some aa to set priors for individual transition distributions p(: | C,B)   
      rng: If a random number generator is given, each transition distribution is drawn from the posterior of the distribution given observed counts and alpha under
           a conjugate categorical-dirichilet model. Otherwise post mean (or MLE if alpha=None or 0) is returned.
             
    Returns:
      transition_probabilities: Array of depth+1 dimensions. Each dimension is of size num_symbols.
        transition_probabilities[C,B,A] = probability of observing C after (B,A) was observed
    """

    if(not np.all(np.array(transition_count_matrix.shape) == transition_count_matrix.shape[0])):
       raise ValueError("dimensions of transition_count_matrix must all be equal")

    transition_probabilities = np.array(transition_count_matrix, dtype=float)
    if(np.isscalar(alpha)):
        transition_probabilities += alpha
    elif(isinstance(alpha,np.ndarray)):
        if(not np.all(np.array(alpha.shape) == transition_count_matrix.shape[0])):
            raise ValueError("dimensions of alpha must all be equal to transition_count_matrix.shape[0]")
        if(alpha.ndim > transition_count_matrix.ndim):
            raise ValueError("alpha.ndim must be less than or equal to transition_count_matrix.ndim")
        
        transition_probabilities += alpha

    elif(alpha is not None):
        raise ValueError("invalid alpha: must be scalar or ndarray with each dimension of size transition_count_matrix.shape[0] and number of dimensions <= the dimensions of transition_count_matrix")

    if(rng is not None):
        s_0 = transition_probabilities.shape
        for ii in np.ndindex(s_0[:-1]):
            transition_probabilities[ii] = rng.dirichlet(transition_probabilities[ii])

    else:
        transition_probabilities = transition_probabilities / transition_probabilities.sum(axis=-1, keepdims=True)

    return transition_probabilities


def get_transition_probabilities_hierarchical(transition_count_matrix : npt.NDArray[np.int_], alpha : float | list = 1, rng : np.random.Generator = None) -> npt.NDArray[np.float_]:
    """ Turns transitions counts into transition probabilities with a cheap, hierchical regularizer.
        Samples from the posterior given the prior are drawn if a random number generator is provides.
        Estimates hierarchically by sequentially increasing the depth of the transition probabilities and using that distribution as a prior
        for the deeper model with scale alpha. First step (no dependency) is estimated with a flat dirichlet prior (single alpha value).

    Args:
      transition_count_matrix: Array of depth+1 dimensions. Each dimension is of size num_symbols.
        transition_count_matrix[C,B,A] = number of times the sequence (C,B,A) was observed
      alpha: concentration parameter of dirichlet prior. alpha can be a scalar or a list of length depth+1 for regularizing at each step
      rng: If a random number generator is given, each transition distribution is drawn from the posterior of the distribution given observed counts and alpha under
           a conjugate categorical-dirichilet model. Distribtions are drawn at each step of the hierarchy. Otherwise post mean (or MLE if alpha=None or 0) is returned.
             
    Returns:
      transition_probabilities: Array of depth+1 dimensions. Each dimension is of size num_symbols.
        transition_probabilities[C,B,A] = probability of observing C after (B,A) was observed
    """

    depth = transition_count_matrix.ndim - 1

    if(np.isscalar(alpha)):
        alpha = np.ones((depth+1))*alpha
    else:
        alpha = np.array(alpha).ravel()
    if(alpha.size < depth + 1):
        raise ValueError("alpha must be list of length depth+1")
    
    transition_probabilities = 1
    for dd in range(depth+1):
        transition_count_matrix_c = transition_count_matrix.sum(axis=tuple(range(0,depth-dd)))
        transition_probabilities = get_transition_probabilities(transition_count_matrix=transition_count_matrix_c, alpha=alpha[dd]*transition_probabilities, rng=rng)

    return transition_probabilities



def simulate_from_model(transition_probabilities : npt.NDArray[np.float_], initial_state : npt.NDArray[np.int_], rng : np.random.Generator, sim_length : int = 100, nsims : int = 1) -> npt.NDArray[np.int_]:
    """
    Args:
      transition_probabilities: Array of depth+1 dimensions. Each dimension is of size num_symbols.
        transition_probabilities[C,B,A] = probability of observing C after (B,A) was observed
      initial_state: integer array of the first 'depth' observations. values must be in range(num_symbols)
            Can be a 1-D array of length depth to be used for all simulations or a 2-D array of size (depth,nsims) for a distinct
            initial state for each simulation.
      rng: The random number generator to use (calls 'choice')
      sim_length: the length of the simulations (including the length of the initial_state)
      nsims: if initial state is 1-D, then is the number of simulations to run (ignored if initial_state is 2 dimensional)

    Returns:
      integer array of sim_length x nsims giving the simulations (initial_state is included in this array)
    """
    
    depth = transition_probabilities.ndim - 1
    num_symbols = transition_probabilities.shape[0]
    if(not np.all(np.array(transition_probabilities.shape) == num_symbols)):
        raise ValueError("invalid shape for transition_probabilities")

    initial_state = np.array(initial_state,dtype=int)
    if(not np.all(np.isin(initial_state,range(num_symbols)))):
        raise ValueError("initial_state must contain only integers in range(num_symbols) where num_symbols=transition_probabilities.shape[0]")

    if(initial_state.ndim > 1):
        nsims = initial_state.shape[1]
    else:
        initial_state = np.tile(initial_state, (nsims,1)).T
    t_0 = initial_state.shape[0]
    
    if(t_0 < depth):
        raise ValueError("initial_state must have at least depth (where depth = transition_probabilities.ndim-1) elements") 
    
    Y = np.zeros((sim_length,nsims),dtype=int)
    Y[:t_0,:] = initial_state

    
    for ss in range(nsims):
        for tt in range(t_0, sim_length):
            sub_seq = tuple(Y[tt-depth:tt,ss])
            Y[tt,ss] = rng.choice(num_symbols, p=transition_probabilities[sub_seq])

    return Y

def get_sequence_likelihood(seq : list | npt.ArrayLike, transition_probabilities : npt.NDArray[np.float_], return_individual : bool = False) -> float | tuple[float, npt.NDArray[np.float_]]:
    depth = transition_probabilities.ndim - 1
    seq = np.array(seq).ravel()

    likelihoods = np.ones((seq.size))
    likelihoods[:depth] = np.nan
    for tt in range(depth, seq.size):
        likelihoods[tt] = transition_probabilities[tuple(seq[tt-depth:tt+1])]

    log_like = np.sum(np.log(likelihoods[depth:]))
    if(return_individual):
        return log_like, likelihoods
    else:
        return log_like
    
def get_sequence_likelihood_with_condition(seq : list | npt.ArrayLike, condition : list | npt.ArrayLike, transition_probabilities : npt.NDArray[np.float_], return_individual : bool = False) -> float | tuple[float, npt.NDArray[np.float_]]:
    depth = transition_probabilities.ndim - 2
    seq = np.array(seq).ravel()
    condition = np.array(condition).ravel()

    likelihoods = np.ones((seq.size))
    likelihoods[:depth] = np.nan
    for tt in range(depth, seq.size):
        likelihoods[tt] = transition_probabilities[(condition[tt-1],) + tuple( seq[tt-depth:tt+1])]

    log_like = np.sum(np.log(likelihoods[depth:]))
    if(return_individual):
        return log_like, likelihoods
    else:
        return log_like
