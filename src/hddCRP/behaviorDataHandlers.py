import numpy as np
from numpy.typing import ArrayLike


EMPTY_INDICATOR = "NULL"

# TODO: Function to take a set of sessions and return a hddCRPModel

# TODO: Function to get tree of groups for time series of actions
def create_context_tree(seq : ArrayLike, depth : int = 3, delim : str = "-") -> np.ndarray:
    depth = int(depth)
    assert depth >= 1, "depth must be positive"
    
    seq = np.array(seq).flatten().astype(str);
    N = len(seq);
    assert N > 0, "seq can't be empty"



    dt = '<U' + str(np.max([len(EMPTY_INDICATOR)] + [len(xx) for xx in seq]) + len(delim))*(depth-1)
    grps = np.zeros((N,depth),dtype=dt)

    append_func = np.vectorize(lambda x,y : x + delim + y)

    for dd in range(depth):
        if(dd > 0):
            for dd2 in range(dd,depth):
                vv = append_func(seq, grps[:,dd2])
                grps[:,dd2] = vv
        seq =  np.roll(seq,1)
        seq[0] = EMPTY_INDICATOR
    return grps

# Also, concatenate over sessions

# TODO: Function to get distances from a set of time series actions (set of sessions)

