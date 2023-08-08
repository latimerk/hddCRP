import numpy as np;

# for typing and validating arguments
from numpy.typing import ArrayLike
from typing import Callable
from inspect import signature

# special functions
from scipy.special import softmax
from scipy.special import logsumexp

import warnings

def exponential_distance_function(f, weights):
    vv = np.isinf(f)
    if(np.all(vv)):
        return 0
    else:
        return np.prod(np.exp(-np.abs(f[~vv])/weights[~vv]))


class hddCRPModel():
    '''
    Holds info for a hierarchical distance dependent Chinese restaurant process (hddCRP) model where we have exact, discrete observations (N observations labeled 1 to M).
    This class helps with Gibbs sampling the process's latent variables given the data, as well as computing log-likelihoods (for Metropolis-Hastings steps over parameters) and predicitve inference.

    The generative model of the hddCRP works is defined via a sampling process.
        The model contains num_layers in a hierarchy where each layer contains N corresponding nodes (for the N observations).
        Each node ("customer" in the CRP, but a node in my visualization) in the final layer corresponds to an observation - but could be
        generalized to have observations at different layers.
        Each node connects to one other node (connection denoted C_{nn,layer}), giving a directed graph with outdegree = 1 for all nodes.
            Self-connections are allowed in the first (most shallow) layer. Otherwise, C_{nn,layer}=nn means the node connects to its counterpart in
            the next higher (layer-1). All other connections are within layer.
        Within each layer, nodes are grouped so that connections only can occur within groups. The group in layer 1 should probably contain all nodes.
            This code right now does not enforce a tree structure on the groupings, but that's what it's intended for.

            for layers ll in {1,...,num_layers}
                for node nn in {1,...,N}
                    sample C_{nn,ll} in {1,...,N}
                        where P(C_{nn,ll}| D, alpha) is proportional to
                            F(D_{nn,tt}) if tt != nn and groups_{ll}(tt) == groups_{ll}(nn)
                            0 if groups_{ll}(tt) != groups_{ll}(nn)
                            alpha_{ll} if tt == nn   -> this is the self-connection, if ll > 1, then the connection points to node nn in layer (ll-1)
                                                        thus, alpha controls how much the high layers influence low layers. With alpha_{ll}->infinity, layer ll becomes a copy of layer (ll-1)

                            F is a function giving positive values and D_{nn,tt} is the distance. F should be decreasing (or constant) with distance

                            F(D_{nn,tt}) is 1 if tt < n and 0 otherwise, everything becomes a set of CRPs: that is, where all the observation probabilities are estimated with a (heirarchical) Dirichlet prior.
                            
                            If alpha_{1:(num_layers-1)} -> infinity (trends toward only self/back connections everywhere but the final layer), then each group in the bottom layer becomes an independent ddCRP

            Then observations are generated:
                Each connected component in the graph is a "table" (each node is a "customer", linked customers sit together)
                Each table is labeled i.i.d. from draws in {1,...,M} with probabilities given in BaseDistribution
                Each node is inherits the table's label
                The observations (the vector Y) are the labels in the final layer
    '''
    UNKNOWN_OBSERVATION = -1;

    def __init__(self, Y : ArrayLike | None, groupings : None | ArrayLike,
                       alpha : float | ArrayLike,
                       D : ArrayLike,
                       Y_values : ArrayLike = None, BaseMeasure : ArrayLike | None = None,
                       weight_params : float | ArrayLike = 1, weight_func : Callable = lambda x, y : np.sum(np.greater(x,0)*y), weight_param_labels : list[str] = None) -> None:
        """
        Sets up a hierarchical distance dependent Chinese restaurant process (hddCRP) model.
        This is a basic model that assumes the observations are from a discrete distribution and fully observed (not a mixture model,
        but still allows for missing data points).

        Generates a valid initial state for starting inference. This uses the current numpy random state.

        Args
          Y: (length N) The observations. Should be a list or numpy vectors. NaN or None values indicate missing data to be inferred by the model.
             IF is None: The constructor will simulate from the hddCRP instead. Y_values must be valid.
          groupings: (array size N x num_layers) The groups of observations at each level. If empty or None, then the model only has 1 layer containing all observations.
          alpha: (length num_layers or scalar) The weight for the self-link or up-layer link the in the hddCRP. Can be per layer (scalar) or different for each layer (list).
          D: (size N x N x K) the distances between each point: can be parameterized by K variables per pair of observations.   
          Y_values: (length M) The possible values Y can take (redundant if all values observed - this is just in case). If None, it contains the unique
                    elements of Y.
          BaseMeasure: (length M or empty) The base measure of observations. If None (or scalar), then assumes uniform distribution over Y_values.
          weight_params: (length P) The parameters for the distance-to-weight function
          weight_func: (length K array - distances, length P array - params) -> non-negative scalar
                       Function for taking a single distance between two observations into the weight for connection from one node to the next in the hddCRP.
          weight_param_labels: (length P) List of names for each of the weight_params values
        
                Example with only four observations:
                 Y = [0, 1, 1, 0]
                 Y_values = [0, 1] # binary data only
                 # or could even be
                 # Y = ['left, 'right', 'right', 'left']
                 # Y_values = ['left', 'right']
                 D = [[ 0, -1, -2, -3]
                      [ 1,  0, -1, -2],
                      [ 2,  1,  0, -1],
                      [ 3,  2,  1,  0]]
                 groupings = [[0, 0],
                              [0, 0],
                              [0, 1],
                              [0, 1]]; # only two levels: behavioral model of interest in developing this will have 3
                 alpha = [1,2] #  will likely want to fit these values


                 weight_func = lambda d,p : np.exp(-np.abs(d)/p)
                 weight_params = 1
                 defines an exponential weight function on the distances between nodes with length scale 1.
                   Likely will want to fit weight_params as a parameter of interest

                 weight_func = lambda d,p : (d > 0)*1.0
                 weight_params = []
                 defines a sequentual CRP that weights everything like a normal CRP
        """

        # sets _Y to a vector of ints. All negative numbers and nans are changed to a single index of UNKNOWN_OBSERVATION
        # makes the N property valid
        if(not Y is None):
            self._Y = np.array(Y).flatten()
            # self._Y[~np.equal(self._Y, self._Y)] = np.nan
            # self._Y[np.isnan(self._Y)] = hddCRPModel.UNKNOWN_OBSERVATION;
            # self._Y = self._Y.astype(int)

            self._simulated_Y = False;


        if(not Y_values is None):
            # If Y_values is given, sets it up as a vector similar to Y.
            self._Y_values = np.array(Y_values).flatten()
        elif(not Y is None):
            self._Y_values = np.unique(self._Y)

        # removes UNKNOWN_OBSERVATION from self._Y_values
        self._Y_values = np.unique(self._Y_values[~np.isin(self._Y_values,[np.nan, None])]);
        # now that self._Y_values is set, M property is valid
        if(Y is None):
            # now that Y_values type is set up, creates space for a simulated Y
            self._simulated_Y = True;
            self._Y = np.zeros((D.shape[0]), dtype=self._Y_values.dtype); # empty Y the shape of the weights
            self._Y.fill(np.nan)

        #checks to see if all values of Y are in Y_values. Note that Y_values does not need to contain UNKNOWN_OBSERVATION/nan
        Y_is_not_nan = np.equal(self._Y, self._Y)#~np.isnan(self._Y);
        assert np.all(np.isin(self._Y[Y_is_not_nan], self._Y_values)), "Inconsistent labels in Y and Y_values found"

        # records values indicies of each type of Y
        self._Y_unknowns = np.where(~Y_is_not_nan)[0];
        self._Y_indicies = [np.where(self._Y == xx)[0] for xx in self._Y_values];

        # sets up groupings: makes num_layers property valid
        if(not groupings is None and len(groupings) >= 1):
            self._groupings = np.array(groupings)
            if(self._groupings.ndim == 1):
                self._groupings = self._groupings[:,np.newaxis];
            if(self._groupings.shape[0] != self.N and self._groupings.shape[1] == self.N):
                self._groupings = self._groupings.T;

            assert self._groupings.ndim == 2, "groupings must be a matrix (ndim == 2)"
            assert self._groupings.shape[0] == self.N, "number of rows of groupings must match number of observations"
        else:
            self._groupings = np.zeros((self.N,1),dtype=int) # default grouping

        #unique groups per each layer: makes property valid
        self._unique_groups = [np.unique(self._groupings[:,xx]) for xx in range(self.num_layers)]
        if(len(self._unique_groups[0]) > 1):
            warnings.warn("Base layer has multiple groups: this is not considered a typical case");


        # indicies for the observations contained in each group at each level
        self._group_indicies = [[np.where(self._groupings[:,ii] == yy)[0]  for yy in xx]
                                    for ii,xx in enumerate(self._unique_groups)]; # depth, group, nodes
        self._groupings_compact = np.zeros((self.N, self.num_layers)) # makes sure numbers for group labels are in an easier to use, compact form (groups 1,3,5 become 0,1,2)
        self._groupings_compact.fill(np.nan)
        for layer, ggs in enumerate(self._group_indicies):
            for gnum, lls in enumerate(ggs):
                assert np.all(np.isnan(self._groupings_compact[lls,layer])), "invalid groupings: node can only belong to one group in each layer"
                self._groupings_compact[lls,layer] = gnum;

                if(layer > 0 and len(np.unique(self._groupings_compact[lls,layer-1])) > 1):
                    warnings.warn("Group " + str(gnum) + " in  layer " + str(layer) + " inherits nodes from multiple groups. This is not considered a typical case: this model is intended for a tree structure of node groups.");
        assert not np.any(np.isnan(self._groupings_compact)), "invalid groupings: all nodes must be assigned to a group in each layer"
        self._groupings_compact = self._groupings_compact.astype(int)

        # sets up self connection weights
        self.alpha = alpha;
        # sets up base measure
        self.BaseMeasure = BaseMeasure;

        # sets up and validates D as a numpy array: makes K property valid
        self._D = np.array(D,dtype=float)
        self._F = np.ones((self.N, self.N))
        assert self._D.shape[0:2] == (self.N, self.N), "first two dimensions of D must be NxN to match observations"
        if(self._D.ndim == 2):
            self._D = self._D[:,:,np.newaxis];
        assert self._D.ndim == 3, "D must have 3 dimensions"
            # this section might not work properly in the extreme case of 1 observation, but you shouldn't be doing inferences of any sort in that case
        self._weights = np.zeros((self.N,self.N)); # the actual weights to be computed from the distances given the parameters and function

        # sets up the weighting function
        assert isinstance(weight_func, Callable), "weight_func must be a function"
        sig = signature(weight_func)
        assert len(sig.parameters) == 2, "weight_func must take two parameters"
        self._weight_func = weight_func;
        assert self._weight_func(self._D[0,0,:], weight_params) >= 0, "weight function may not produce valid results" # tries running the weight function to make sure it works

        # sets up weighting function parameters: makes P, weight_params properties valid
        if(weight_params is None):
            weight_params = [];
        self._weight_params = np.array(weight_params);
        self.weight_params = np.array(weight_params);

        if(weight_param_labels is None):
            self._weight_param_labels = ["p" + str(xx) for xx in range(self.P)];
        else:
            assert len(weight_param_labels) == self.P, "incorrect number of weight_param_labels"
            self._weight_param_labels = weight_param_labels;

        # now generates a random valid graph and labels tables for the initial point
        if(not self._simulated_Y):
            self.generate_random_connection_state();
        else:
            self.simulate_from_hddCRP();
    
        # check to make sure everything is happy
        # self._validate_internals("INITIAL SETUP TEST")
    '''
    ==========================================================================================================================================
    Properties of the model.
    Includes model dimensions and variables that can be changed (alpha and weight_params)
    ==========================================================================================================================================
    '''

    @property
    def weight_param_labels(self) -> list[str]:
        return self._weight_param_labels;

    @property
    def N(self) -> int:
        '''
        The number of observations/nodes
        '''
        return self._Y.size;

    @property
    def M(self) -> int:
        '''
        number of observation types (how many values Y can take)
        '''
        return self._Y_values.size;

    
    @property
    def P(self) -> int:
        '''
        number of weight parameters
        '''
        return self._weight_params.size;

    
    @property
    def K(self) -> int:
        '''
        number of distance coefficients per pair of nodes/observations
        '''
        return self._D.shape[2];

    
    @property
    def num_groups(self) -> ArrayLike:
        '''
        number of node groupings in each layer
        '''
        return [len(xx) for xx in self._unique_groups];

    @property
    def num_layers(self) -> int:
        '''
        number of layers in the hierarchy of hddCRP. This is currently determined by the _groupings variable and is immutable
        '''
        return self._groupings.shape[1];

    @property
    def weight_params(self) -> float | ArrayLike:
        '''
        The weight parameters for the node distance function
        '''
        return self._weight_params;
    @weight_params.setter
    def weight_params(self, theta : float | ArrayLike) -> None:
        '''
        Changes the current weight parameters. Updates the self._F weights automatically when called.

        Args
          theta: (length P) the new parameters
        '''
        theta = np.array(theta,dtype=float);
        assert theta.shape == self._weight_params.shape, "parameters must be length (P)"
        self._weight_params = np.array(theta,dtype=float); 
        self._F = self._compute_weights(self._weight_params)
        assert np.all(self._F >= 0), "invalid connection weights found! check the weight/distance function"


    # the alpha values: weight of self/upwards connections for each layer
    @property
    def alpha(self) -> ArrayLike:
        return self._alpha;
    @alpha.setter
    def alpha(self, aa : float | ArrayLike) -> None:
        if(np.size(aa) == 1):
            aa = aa*np.ones((self.num_layers))
        aa = np.array(aa).flatten();
        assert aa.size == self.num_layers, "alpha must have values for all layers (or a scalar)"
        self._alpha = aa;

    @property
    def BaseMeasure(self) -> np.ndarray:
        '''
        The base measure for observation values: typically won't change and the default value is uniform.
        In a single-layer model, the prior over a single observation would be Dirichlet(alpha[0]*BaseMeasure)
        '''
        return self._BaseMeasure;
    @BaseMeasure.setter
    def BaseMeasure(self, BaseMeasure_new : float | ArrayLike | None):
        if(BaseMeasure_new is None):
            BaseMeasure_new = 1;
        if(np.isscalar(BaseMeasure_new)):
            BaseMeasure_new = BaseMeasure_new*np.ones((self.M));
        BaseMeasure_new = np.array(BaseMeasure_new).flatten();
        assert len(BaseMeasure_new) == self.M, "BaseMeasure is incorrect length"
        self._BaseMeasure = BaseMeasure_new/np.sum(BaseMeasure_new);
        
    @property
    def num_parameters(self) -> int:
        '''
        The number of parameters that we probably care about: the distance/weight function parameters and the alphas
        '''
        return self.alpha.size + self.weight_params.size

    '''
    ==========================================================================================================================================
    Some stats about the data
    ==========================================================================================================================================
    '''
    def group_frequencies(self) -> list:
        return [[{yy : np.mean(self._Y[self._group_indicies[layer][gg]] == yy) for yy in self._Y_values}
                    for gg in self.num_groups[layer]]
                    for layer in self.num_layers];
    def group_counts(self) -> list:
        return [[{yy : np.sum(self._Y[self._group_indicies[layer][gg]] == yy) for yy in self._Y_values}
                    for gg in self.num_groups[layer]]
                    for layer in self.num_layers];

    '''
    ==========================================================================================================================================
    Functions for generating a valid initial state
    ==========================================================================================================================================
    '''

    def simulate_from_hddCRP(self) -> None:
        '''
        Generates a simulation from the hddCRP

        Result:
          Y complete overridden with new observations
          This sets up all internal variables that start with _C_
        '''

        # resets Y and Y indicies
        self._Y[:] = np.nan;
        self._Y_unknowns = np.where(self._Y == np.nan)[0];
        self._Y_indicies = [np.where(self._Y == xx)[0] for xx in self._Y_values];

        self._initialize_connections(); # with everything set to unknown, this will just generate connections from the hddCRP
        self._initialize_predecessors();

        self._initialize_table_labels();
        self._initialize_table_cycles();
        self._initialize_redraw_observations();

        # now sets these up properly
        self._Y_unknowns = np.where(self._Y == np.nan)[0];
        self._Y_indicies = [np.where(self._Y == xx)[0] for xx in self._Y_values];


    def generate_random_connection_state(self, DEBUG_STEPS : bool = False) -> None:
        '''
        Generates internal variables for all of the connection/table information.
        The connections between nodes in the hddCRP are generated stochastically (they are unobserved and this assumes we'll do Gibbs sampling later)
        Uses current np.random state

        This sets up all internal variables that start with _C_
        '''
        # label the tables
        self._initialize_connections();
        self._initialize_predecessors();
        
        if(DEBUG_STEPS):
            self._validate_connections()
        
        self._initialize_table_labels();
        
        if(DEBUG_STEPS):
            self._validate_labels()

        self._initialize_table_cycles();
        
        if(DEBUG_STEPS):
            self._validate_cycles()

        self._initialize_table_label_counts();
    
        if(DEBUG_STEPS):
            self._validate_counts()


    def _blank_connection_variables(self) -> None:
        '''
        Sets up the connection variable representations: ddCRP defines directed graph of node with outdegree fixed to 1.
        May uses redundant representations for computationalconvenience. e.g., uses the fact that every component in the graph 
            will contain one and only one cycle (cycle may just be a single node with a self connection).
            The cycle will be contained within one layer
        Everything here will be blank and not valid!

        Assumes: valid properties: self.N, self.num_layers 
                 also setup by constructor: _Y_indicies

        Post: empty: _C_ptr: (int array size N x num_layers) _C_ptr[n,l] is connection value (node number) that node n in layer l connects to 
                     _C_is_cycle: (bool array size N x num_layers)
                     _C_predecessors (list of length num_layers of lists length N of empty lists) -> yes, three layers of indexing!
                     _C_tables (int array size N x num_layers)
                     _C_table_values (int list empty)
                     _C_y (int array size N x num_layers)
                     _C_num_labeled_in_table (empty list of ints)
                     _C_num_labeled_upstream (int array size N x num_layers)
              valid: _C_y_0 (int array size N x num_layers) contains the truly observation value of all nodes in all layers (will be UNKNOWN_OBSERVATION in shallow layters)

        '''
        #self._C_matrix = np.zeros((self.N,self.N,self.num_layers),dtype=bool)# row,col,depth=True means node row connects to node col at depth
        self._C_ptr = -1*np.ones((self.N,self.num_layers),dtype=int); #C_ptr[row,depth] says the node that node row at depth layer connects to
        self._C_is_cycle = np.zeros((self.N,self.num_layers),dtype=bool); # if each node with within the "cycle" portion of the table: helper variable for computations
                                                                          # note: in this digraph, each component can (and must) contain one cycle: everything else points into the cycle
        self._C_predecessors = [[[] for nn in range(self.N)] for ll in range(self.num_layers)] # look up table to get lists of node with arrows pointing TO each node
                            # only conncerned about nodes WITHIN each layer: used to make one step faster, accessed: self._C_predecessors[layer][node]
        self._C_tables = np.zeros((self.N,self.num_layers),dtype=int); #table numbers for each node in the ddCRP (a connected component of the digraph)

        self._C_table_values = np.zeros((0),dtype=int) # observation type for each table: can be UNKNOWN_OBSERVATION
        self._C_table_counter = 0; # number of tables in use

        self._C_num_labeled_in_table = np.zeros((0),dtype=int) # number of nodes with observations in each table (nodes in final layer that aren't UNKNOWN_OBSERVATION)
        self._C_num_labeled_upstream = np.zeros((self.N, self.num_layers),dtype=int) # number of observed nodes upstream of each node (inclusive of that node)

        # creates Y for each layer
        # here, Y uses condensed observation indexes: should be 0,..,(M-1) plus any UNKNOWN_OBSERVATION values
        self._C_y_0 = np.zeros((self.N, self.num_layers), dtype=int)
        self._C_y_0.fill(hddCRPModel.UNKNOWN_OBSERVATION)
        for ii,jj in enumerate(self._Y_indicies):
            self._C_y_0[jj,-1] = ii;
        self._C_y_0[self._Y_unknowns,-1] = hddCRPModel.UNKNOWN_OBSERVATION
        self._C_y_0[:,0:-1] = hddCRPModel.UNKNOWN_OBSERVATION
        self._C_y = np.zeros_like(self._C_y_0);
        self._C_y.fill(hddCRPModel.UNKNOWN_OBSERVATION);
    
        self._CB_y = self._C_y.copy()
        self._CB_is_cycle = self._C_is_cycle.copy()
        self._CB_ptr = self._C_ptr.copy()
        self._CB_table_counter = self._C_table_counter
        self._CB_tables = self._C_tables.copy()
        self._CB_table_values = self._C_table_values.copy()
        self._CB_predecessors = self._C_predecessors.copy()
        self._CB_num_labeled_in_table = self._C_num_labeled_in_table.copy()
        self._CB_num_labeled_upstream = self._C_num_labeled_upstream.copy()

    def _initialize_connections(self) -> None:
        '''
        Generates the connections between nodes. Does not set up tables.
        Uses current np.random state

        Assumes: valid properties: self.N, self.num_layers, alpha,
                 also setup by constructor: _Y_indicies, _group_indicies, _F
              empty:
                    _C_ptr: (int array size N x num_layers) 
              valid:
                    _C_y_0 (int array size N x num_layers) (int array size N x num_layers) contains the truly observation value of all nodes in all layers (will be UNKNOWN_OBSERVATION in shallow layters)

        Post: valid: _C_ptr: (int array size N x num_layers) _C_ptr[n,l] is connection value (node number) that node n in layer l connects to 
                      (list of length num_layers of lists length N of int lists) _C_predecessors[l][n] is list of nodes in layer l (only layer l!) pointing to node n in layer l
                     _C_y_0 (unchanged)
              incomplete:
                    _C_y (int array size N x num_layers) (int array size N x num_layers) contains the observation values of all nodes in all layers (can be UNKNOWN_OBSERVATION)
                                                                                         conditioned on table (if node is unknown in _C_y_0 but connect to an observed value in _C_y, it inherits the connected node's value)
              unchanged: 
                     _C_predecessors
                     _C_is_cycle
                     _C_tables
                     _C_table_values
                     _C_num_labeled_in_table (empty list of ints)
                     _C_num_labeled_upstream (int array size N x num_layers)

        '''
        self._blank_connection_variables();

        # Connections generated from lowest-depth first (where observations are)
        self._C_y = self._C_y_0.copy()
        for layer in range(self.num_layers-1, -1, -1): # for each group
            cs, YY_upper = self._initialize_single_layer_of_connections(self._group_indicies[layer], self._C_y[:,layer], self.alpha[layer]);
            if(layer > 0):
                self._C_y[:,layer - 1] = YY_upper;
            self._C_ptr[:,layer] = cs;

    def _initialize_predecessors(self):
        '''
        Generates the connections between nodes. Does not set up tables.
        Uses current np.random state

        Assumes: valid properties: self.N, self.num_layers
                    _C_ptr: (int array size N x num_layers) _C_ptr[n,l] is connection value (node number) that node n in layer l connects to 
              empty:
                    _C_predecessors (list of length num_layers of lists length N of empty lists) -> yes, three layers of indexing!

        Post: valid: _C_predecessors (list of length num_layers of lists length N of int lists) _C_predecessors[l][n] is list of nodes in layer l (only layer l!) pointing to node n in layer l
                     _C_y_0 (unchanged)
                     _C_ptr (unchanged)
              incomplete:
                    _C_y (int array size N x num_layers) (int array size N x num_layers) contains the observation values of all nodes in all layers (can be UNKNOWN_OBSERVATION)
                                                                                         conditioned on table (if node is unknown in _C_y_0 but connect to an observed value in _C_y, it inherits the connected node's value)
              unchanged: 
                     _C_is_cycle
                     _C_tables
                     _C_table_values
                     _C_num_labeled_in_table (empty list of ints)
                     _C_num_labeled_upstream (int array size N x num_layers)

        '''
        for layer in range(self.num_layers-1, -1, -1):
            for node_from in range(self.N):
                node_to = self._C_ptr[node_from,layer];
                if(node_to != node_from): # no self connections
                    self._C_predecessors[layer][node_to].append(node_from);


    def _initialize_single_layer_of_connections(self, groups : tuple, YY : ArrayLike, alpha : float) -> tuple[ArrayLike,ArrayLike]:
        '''
        Generates connections within a single layer
        Uses current np.random state

        Assumes: valid properties: self.N
                 also setup by constructor: _F
        Results: no changes in this function to the connection variables

        Args:
          groups: each element of the tuple is a list of the members in a group in the current layer
          YY: (length N) the labels of the node in the current layer. Can contain UNKNOWN_OBSERVATION values. With pass-by-reference in numpy,
              UNKNOWN_OBSERVATION values may be replaced if nodes are connected to labeled nodes.
          alpha: the self connection weight for the layer

        Returns:
          (connections, YY_upper): int arrays each of length N.
                                   connections: label of node each node connects to
                                   YY_upper: known labels to propogate to layer above
        '''
        YY_upper = np.ones((self.N),dtype=int)
        YY_upper.fill(hddCRPModel.UNKNOWN_OBSERVATION)
        connections = np.ones((self.N),dtype=int)
        connections.fill(-1)

        # for each group
        for gg in groups:
            # for each node in group
            for node in gg:
                # if is labeled:
                if(YY[node] != hddCRPModel.UNKNOWN_OBSERVATION):
                    # can connect to same label nodes or unknown label nodes
                    can_connect_to = gg[(YY[gg] == YY[node]) | (YY[gg] == hddCRPModel.UNKNOWN_OBSERVATION)];
                # if isn't labeled
                else:
                    # can connect to all in group
                    can_connect_to = gg;

                # get weights of weight is possible to connect to
                ps = self._F[node, can_connect_to];
                ps[can_connect_to == node] = alpha;
                ps = ps/ps.sum()

                #generate a connection
                connections[node] = can_connect_to[np.random.choice(can_connect_to.size, p=ps)];

                # if connecting to unlabeled node and is labeled, adds label
                if(YY[node] != hddCRPModel.UNKNOWN_OBSERVATION and YY[connections[node]] == hddCRPModel.UNKNOWN_OBSERVATION):
                    ff = node;
                    while(connections[ff] >= 0 and YY[connections[ff]] == hddCRPModel.UNKNOWN_OBSERVATION):
                        YY[connections[ff]] = YY[node];
                        ff = connections[ff];
                        # if reached a connection to above layer
                        if(connections[ff] == ff):
                            YY_upper[ff] = YY[node];

                # if connecting to labeled node and is unlabeled, adds label
                if(YY[node] == hddCRPModel.UNKNOWN_OBSERVATION and YY[connections[node]] != hddCRPModel.UNKNOWN_OBSERVATION):
                    Y_new = YY[connections[node]];
                    YY[node] = Y_new;

                    # must track connections backwards
                    prevs = np.where(connections == node)[0]
                    while(len(prevs) > 0):
                        prev_c = prevs[0];
                        YY[prev_c] = Y_new;
                        prevs = np.delete(prevs,0);

                        prevs = np.append(prevs, np.where(connections==prev_c)[0])

                # if connects to above layer, adds label to upper layer
                if(connections[node] == node):
                    YY_upper[node] = YY[node];

        return (connections, YY_upper)


    def _initialize_table_labels(self) -> None:
        '''
        Labels all the tables given the current connections. _C_y will be filled in for all tables.

        Assumes: valid properties: self.N, self.num_layers
                 also setup by constructor: _Y_indicies, _group_indicies, _F
              valid: _C_ptr: (int array size N x num_layers) _C_ptr[n,l] is connection value (node number) that node n in layer l connects to 
                     _C_predecessors (list of length num_layers of lists length N of int lists) _C_predecessors[l][n] is list of nodes in layer l (only layer l!) pointing to node n in layer l
                     _C_y_0
              incomplete or empty:
                     _C_y (int array size N x num_layers) (int array size N x num_layers) 
              empty: 
                     _C_is_cycle: (bool array size N x num_layers)
                     _C_tables (int list empty)
                     _C_table_values (empty list of ints)
                     _C_num_labeled_in_table (empty list of ints)
                     _C_num_labeled_upstream (int array size N x num_layers)

        Results:
              valid:
                     _C_y (int array size N x num_layers) (int array size N x num_layers) contains the observation values of all nodes in all layers (can be UNKNOWN_OBSERVATION)
                                                                                         conditioned on table (if node is unknown in _C_y_0 but connect to an observed value in _C_y, it inherits the connected node's value)
                     _C_table_counter: (int) current table index
                     _C_tables (int array size N x num_layers) table numbers for each observation
                     _C_table_values (int array length _C_table_counter) observation value for each table
                     _C_ptr  (unchanged)
                     _C_predecessors (unchanged)
              unchanged:
                     _C_is_cycle
              empty:_C_num_labeled_in_table (int array )
                    _C_num_labeled_upstream (int array size N x num_layers)

        Raises:
          RuntimeError: if invalid table connections are found (different observations connected in ddCRP)
        '''
        self._C_table_counter = 0;
        self._C_table_values = np.zeros((0),dtype=int)
        self._C_tables.fill(-1);

        counted = np.zeros((self.N, self.num_layers), dtype=bool);
        for layer in range(self.num_layers): # here, the order of layer transversal doesn't matter
            for ii in range(self.N):
                if(not counted[ii, layer]):
                    # start with only current node and layer
                    connected_nodes = [(ii,layer)];
                    #while any nodes left to analyze
                    while(len(connected_nodes) > 0):
                        node_c, layer_c = connected_nodes[0];
                        del(connected_nodes[0])
                        # if node hasn't been touched yet
                        if(not counted[node_c,layer_c]):
                            # labels the table
                            self._C_tables[node_c,layer_c] = self._C_table_counter;
                            counted[node_c,layer_c] = True;

                            # finds any nodes connected to current node (forward & backward connections) and adds to list
                            for kk in self._C_predecessors[layer_c][node_c]:
                                connected_nodes += [(kk,layer_c)]

                            # if node in lower layer is connected
                            if(layer_c < self.num_layers - 1 and self._C_ptr[node_c,layer_c+1] == node_c):
                                connected_nodes += [(node_c,layer_c+1)]
                            next_node,next_layer = self._get_next_node(node_c, layer_c);
                            connected_nodes += [(next_node,next_layer)]
                    # finds the table label (connection generator code doesn't necessarily label everything: may be mix of labeled & unknown)
                    labels = np.unique(self._C_y_0[self._C_tables == self._C_table_counter]);
                    labels = labels[labels != hddCRPModel.UNKNOWN_OBSERVATION];
                    if(labels.size > 1):
                        raise RuntimeError("Invalid table setup acheived during initialization: shouldn't happen!")
                        # observed nodes cannot have multiple labels at same table
                    elif(labels.size == 1):
                        label_c = labels[0];
                    else:
                        label_c = hddCRPModel.UNKNOWN_OBSERVATION

                    # stores the table label
                    self._C_y[self._C_tables == self._C_table_counter] = label_c;
                    self._C_table_values = np.append(self._C_table_values, label_c);
                    
                    self._C_table_counter += 1;

    def _initialize_table_label_counts(self) -> None:
        '''
        Counts how many labeled nodes in tables, and upstream of each node.

        Assumes: valid properties: self.N, self.num_layers
              valid: _C_ptr: (int array size N x num_layers) _C_ptr[n,l] is connection value (node number) that node n in layer l connects to 
                     _C_y_0
                     _C_tables (int list )
                     _C_y (unused)
                     _C_ptr  (unused)
                     _C_table_counter (unused)
                     _C_tables (unused)
                     _C_is_cycle
              empty: 
                     _C_num_labeled_in_table (empty list of ints)
                     _C_num_labeled_upstream (int array size N x num_layers)

        Results:
              valid:
                     _C_num_labeled_in_table (int array length _C_table_counter) number of observed label nodes at each table
                     _C_num_labeled_upstream (int array size N x num_layers) number of observed label nodes upstream of each node in the graph (inclusive of the node) Only needs to be valid for nodes where _C_is_cycle is False
                     _C_y_0 (unchanged)
                     _C_ptr  (unchanged)
                     _C_table_counter (unchanged)
                     _C_tables (unchanged) 
              unchanged:
                     _C_y 
                     _C_table_values 
                     _C_predecessors 
        '''
        # For each table, counts number of labeled nodes. Precomputing is useful for fast Gibbs computations.
        self._C_num_labeled_in_table = np.array([np.sum(self._C_y_0[self._C_tables == tt] != hddCRPModel.UNKNOWN_OBSERVATION) 
                                                for tt in range(self._C_table_counter)], dtype=int)

        # trace all labeled nodes
        labeled_coords = np.argwhere(self._C_y_0 != hddCRPModel.UNKNOWN_OBSERVATION);
        self._C_num_labeled_upstream.fill(0)
        for pp in labeled_coords:
            node = pp[0];
            layer = pp[1];

            while(True): # do while
                self._C_num_labeled_upstream[node,layer] += 1;

                if(not self._C_is_cycle[node,layer]): # while
                    node, layer = self._get_next_node(node, layer);
                else:
                    break;
                
                

    def _initialize_redraw_observations(self) -> None:
        '''
        Redraws table labels from base distribution

        Assumes: all _C valid (except  _C_num_labeled_upstream and _C_num_labeled_in_table)
        Results:
            _C_y_0, _C_y, and C_table_values redrawn using the base measure
            _C_num_labeled_upstream and _C_num_labeled_in_table updated.

            all _C should be valid
        '''
        # goes through each node in final layer
        for nn in range(self.N):
            #if unknown, draws label and sets label to entire table 
            if(self._C_y_0[nn,-1] == hddCRPModel.UNKNOWN_OBSERVATION):
                table_num = self._C_tables[nn,-1]# table num
                table_val = np.random.choice(self.M, self.BaseMeasure)

                self._C_table_values[table_num] = table_val

                # set all observations (final layer only) at that table
                self._C_y_0[self._C_tables[:,-1] ==  table_num,-1] = table_val;
                self._Y[self._C_tables[:,-1] ==  table_num] = self._Y_values[table_val];

                # set all nodes at that table
                self._C_y[self._C_tables ==  table_num] = table_val;


        self._initialize_table_label_counts();

    def _initialize_table_cycles(self) -> None:
        '''
        For each table, figures out which nodes are part of the cycle that must be present in the digraph. Useful data for Gibbs sampling steps

        Assumes: valid properties: self.N, self.num_layers
              valid: _C_ptr: (int array size N x num_layers) _C_ptr[n,l] is connection value (node number) that node n in layer l connects to 
                     _C_tables
                     _C_table_counter
              empty: 
                     _C_is_cycle: (bool array size N x num_layers)

        Result:
              valid:
                     _C_is_cycle: (bool array size N x num_layers) if each node in each layer is part of it's current table's cycle
                     _C_ptr (unchanged)
                     _C_table_counter (unchanged)
                     _C_tables  (unchanged)
              unchanged:
                     _C_y 
                     _C_table_values 
                     _C_num_labeled_in_table 
                     _C_num_labeled_upstream 
                     _C_predecessors
        '''
        self._C_is_cycle[:,:] = False
        for tt in range(self._C_table_counter):
            # visit count
            visits = np.zeros((self.N, self.num_layers),dtype=int);

            # get first node in table (doesn't matter where we start)
            node, layer = np.argwhere(self._C_tables == tt)[0]
            visits[node,layer] += 1;
            while(visits[node,layer] < 2):
                node, layer = self._get_next_node(node, layer);
                visits[node,layer] += 1;

            # now from current point, labels full cycle
            while(not self._C_is_cycle[node,layer]):
                self._C_is_cycle[node,layer] = True;
                node, layer = self._get_next_node(node, layer);



    '''
    ==========================================================================================================================================
    Gibbs sampling functions: resamples connections one at a time given the remaining connections.
    Functions can swap connections and relable tables.
    ==========================================================================================================================================
    '''
    def run_gibbs_sweep(self, order : ArrayLike = None, DEBUG_MODE : bool = False) -> np.ndarray:
        '''
        Gibbs sampling for each node in the model. Each connection for a node is sampled one at a time (sequentially) from the posterior distribution 
        over connections given all the other connections (and the parameters: alpha, weight_params, and BaseMeasure).

        Uses current np.random state

        Args:
          order: ((N*num_layers) x 2) Order of nodes to sample. Each row is a (node,layer) index.
                 By default, does all nodes in each layer from layer 0 to num_layers-1.
                 Within each layer, node order 0 to (N-1).
        Returns:
          Set of internal debug codes for each gibbs step
        '''
        if(order is None):
            order = np.concatenate((np.tile(np.arange(self.N),(self.num_layers))[:, np.newaxis],
                                    np.tile(np.arange(self.num_layers)[:,np.newaxis],(1,self.N)).flatten()[:,np.newaxis]), axis=1)

        codes = np.zeros(order.shape, dtype=int)
        for ii in range(order.shape[0]):
            codes[order[ii,0], order[ii,1]] = self._gibbs_sample_single_node(order[ii,0], order[ii,1], DEBUG_MODE=DEBUG_MODE);
        return codes

    def _post_for_single_nodes_connections(self, node : int, layer : int ) -> tuple[list,np.ndarray]:
        '''
        Gets the posterior probability for each possible connection for one node given all the other connections 

        Args:
          node: int in range(self.N)
          layer: int in range(self.num_layers)

        Returns:
          tuple(can_connect_to, log_post)
            can_connect_to (int list, length R) all nodes (just node numbers, layer is implied) of possible connections
            post (array length R) the log posterior probability of node connecting to each node in can_connect_to
                                      up to a constant
        '''

        # find nodes that node can connect to: must match group & observation label
        group_num = self._groupings_compact[node,layer];
        table_num = self._C_tables[node,layer];
        Y_type = self._C_y[node,layer]

        can_connect_to = self._group_indicies[layer][group_num]; # in group
        can_connect_to = can_connect_to[np.isin(self._C_y[can_connect_to,layer], [hddCRPModel.UNKNOWN_OBSERVATION, Y_type])]; # possible observation
        if(layer > 0 and not np.isin(self._C_y[node,layer-1], [hddCRPModel.UNKNOWN_OBSERVATION, Y_type])): # connection to upper layers must also have correct label
            can_connect_to = can_connect_to[can_connect_to != node];

        assert can_connect_to.size > 0, "node has no possible connections"

        # get table info for all possible connections
        table_count_delta = np.zeros((can_connect_to.size),dtype=int);

        destination_tables = self._C_tables[can_connect_to,layer];
        if(layer > 1):
            destination_tables[can_connect_to == node] = self._C_tables[node,layer-1];
        out_of_table = destination_tables != table_num;

        destination_Y = self._C_table_values[destination_tables];

        # probabilities of observing each table count that a move is associated with (connecting UNKNOWN_OBSERVATION to a known table will determine table type)
        # log_G_delta = np.zeros((can_connect_to.size));
        # if(Y_type != hddCRPModel.UNKNOWN_OBSERVATION):
        #     log_G_delta[:] = self.log_BaseMeasure[Y_type]
        # log_G_delta[destination_Y != hddCRPModel.UNKNOWN_OBSERVATION] = self.log_BaseMeasure[destination_Y[destination_Y != hddCRPModel.UNKNOWN_OBSERVATION]];
        G_delta = np.zeros((can_connect_to.size));
        if(Y_type != hddCRPModel.UNKNOWN_OBSERVATION):
            G_delta[:] = self.BaseMeasure[Y_type]
        G_delta[destination_Y != hddCRPModel.UNKNOWN_OBSERVATION] = self.BaseMeasure[destination_Y[destination_Y != hddCRPModel.UNKNOWN_OBSERVATION]];


        # label each node as splitting, combining, or the same
        if(self._C_is_cycle[node,layer]):
            #if is in cycle, connecting node to out of table will combine tables
                # for combining, will it change number of labeled nodes?
                    # if t1 or t2 are unlabeled: no change in counts
                    # if both labeled, subtracts 1 from count
            if(Y_type != hddCRPModel.UNKNOWN_OBSERVATION):
                table_count_delta[out_of_table & destination_Y != hddCRPModel.UNKNOWN_OBSERVATION] = -1;
        else:
            has_labeled_nodes_remaining = (self._C_num_labeled_in_table[table_num] - self._C_num_labeled_upstream[node, layer]) > 0; # any initially labeled nodes in table that don't point towards the current node
            if(has_labeled_nodes_remaining): # only need to compute follow if this is true
                upstream_nodes = self._get_preceeding_nodes_within_layer(node, layer); # nodes with connections leading to current node (or current node!)
                is_labeled_upstream = self._C_num_labeled_upstream[node, layer] > 0;

            #if not in cycle, connecting node to out of table add to other table
                # for partial combining, will it change number of labeled nodes?
                    # if current node labeled (something upstream AND downstream of node labeled) and t2 not labeled, will add one
                    # otherwise, no change
            if(has_labeled_nodes_remaining and is_labeled_upstream):
                table_count_delta[out_of_table & destination_Y == hddCRPModel.UNKNOWN_OBSERVATION] += 1;

            # if is not in the cycle, changing connection to a preceding node will split the table
            if(has_labeled_nodes_remaining and is_labeled_upstream):
                # for splitting, will it change number of labeled nodes?
                    # if both new tables contain labeled nodes, adds one
                    # otherwise, stays the same
                table_count_delta[np.isin(can_connect_to, upstream_nodes)] += 1;

        # probability (up to a constant) of P(table observation values | connections)
        # log_p_Y = log_G_delta * table_count_delta;
        p_Y = G_delta ** table_count_delta;

        # gets the log prior probability of each connection (up to a constant)
        p_C = self._F[node,can_connect_to]
        p_C[can_connect_to == node] = self.alpha[layer];
        #log_p_C = np.log(p_C);

        # gets log posterior probability
        # log_post  = log_p_Y + log_p_C 
        # #log_post -= logsumexp(log_post);
        # return (can_connect_to, log_post)

        # gets  posterior probability
        post = p_Y * p_C
        return (can_connect_to, post)

    def _gibbs_sample_single_node(self, node : int, layer : int, DEBUG_MODE : bool = False) -> int:
        '''
        Samples from posterior probability for each possible connection for one node given all the other connections and observations (and alpha, weight_params, and BaseMeasure)

        Args:
          node: int in range(self.N)
          layer: int in range(self.num_layers)
        Returns:
          (int) internal debug code for type of connection change (new table etc.)

        Results:
          Connection/table values can change
        '''
        # can_connect_to,log_post = self._log_post_for_single_nodes_connections(node, layer);
        # # gumbels = np.random.gumbel(size=can_connect_to.size)
        # # node_to = can_connect_to[np.argmax(log_post + gumbels)]
        # node_to = np.random.choice(can_connect_to, softmax(log_post))
        
        if(DEBUG_MODE):
            self._CB_y = self._C_y.copy()
            self._CB_ptr = self._C_ptr.copy()
            self._CB_is_cycle = self._C_is_cycle.copy()
            self._CB_table_counter = self._C_table_counter
            self._CB_tables = self._C_tables.copy()
            self._CB_table_values = self._C_table_values.copy()
            self._CB_predecessors = self._C_predecessors.copy()
            self._CB_num_labeled_in_table = self._C_num_labeled_in_table.copy()
            self._CB_num_labeled_upstream = self._C_num_labeled_upstream.copy()

        can_connect_to, post = self._post_for_single_nodes_connections(node, layer);
        node_to = np.random.choice(can_connect_to, p = post/np.sum(post))

        connection_type = self._set_connection(node, node_to, layer)
        if(DEBUG_MODE):
            self._validate_internals(connection_type)
            print("Connection type: " + str(connection_type))
        return connection_type

    '''
    ==========================================================================================================================================
    Methods for model fitting & evaluation
    ==========================================================================================================================================
    '''

    def compute_log_likelihood(self, weight_params : ArrayLike = None, alphas : ArrayLike | float = None):
        '''
        Computes the log likelihood of the current connections between nodes (customers) in the model and observation values (table labels) given the parameters.
        Assumes connection variables are all setup correctly - does not check! No values in object are modified.

        Args:
          weight_params: (array length P) The parameters for the distance/weight function. If none given, uses current values.
          alphas: (array length num_layers or scalar) The level-up/self connection bias parameters. If scalar, uses the same value for all layers. If none given, uses current values.

        Results:
          log likelihood of the current connections and observation types (table labels)
        '''


        (log_P_cons, log_C) = self._compute_log_likelihood_connection(weight_params, alphas)
        log_p_Y_given_cons  = self._compute_log_P_table_assignments()

        return log_P_cons + log_p_Y_given_cons - log_C;


    def _compute_log_likelihood_connection(self, alphas : float | ArrayLike  | None, weight_params : ArrayLike  | None) -> tuple[float,float]:
        '''
        Computes the log likelihood of the current connections between nodes (customers).
        Assumes connection variables are all setup correctly - does not check! No values in object are modified.

        Args:
          weight_params: (array length P) The parameters for the distance/weight function. If none given, uses current values.
          alphas: (array length num_layers or scalar) The level-up/self connection bias parameters. If scalar, uses the same value for all layers. If none given, uses current values.

        Results:
          (log likelihood of connections - unnormalized, log normalizing constant)
        '''

        # set up alphas
        if(alphas is None):
            alphas = self.alpha
        else:
            if(np.isscalar(alphas)):
                alphas = np.ones((self.num_layers))*alphas;
            else:
                alphas = np.array(alphas).flatten()
                assert alphas.size == self.alpha.size, "invalid size of alpha argument: must be scalar or array of length num_layers"

        # get connection weights
        if(weight_params is None): 
            weights = self._F
        else:
            weight_params = np.array(weight_params);
            assert weight_params.size == self._weight_params.size, "weight_params is not the correct size"
            weights = self._compute_weights(weight_params);

        # gets normalizing constants for connections
        log_C = 0;
        for layer in range(self.num_layers):
            for gg in self._group_indicies[layer]:
                log_C += np.sum(np.log(alphas[layer] + np.sum(weights[gg,gg],axis=1)[:,np.newaxis]))

        # gets log likehood of picked weights
        log_P_cons = 0;
        for layer in range(self.num_layers):
            w_c = weights[np.r_[range(self.N)], np.r_[self._C_ptr[:,layer]]];
            w_c[self._C_ptr[:,layer] == range(self.N)] = alphas[layer];
            log_P_cons += np.sum(np.log(w_c))

        return (log_P_cons, log_C)

    def _compute_log_P_table_assignments(self, LogBaseMeasure : ArrayLike = None) -> float:
        '''
        Computes the log likelihood of observed table values
        Assumes connection variables are all setup correctly - does not check! No values in object are modified.

        Args:
          LogBaseMeasyre: (array length M) The log base probability of each observation value. If none given, uses current values.

        Results:
          log likelihood of the observations occuring at each table.
        '''

        # get connection weights
        if(LogBaseMeasure is None): 
            LogBaseMeasure = np.log(self.BaseMeasure)
        else:
            LogBaseMeasure = np.array(LogBaseMeasure).flatten()
            assert LogBaseMeasure.size == self.M, "Base measure is not the correct size"
            LogBaseMeasure -= logsumexp(LogBaseMeasure)

        # gets log likelihood of table assignments
        num_tables = np.array([np.sum(self._C_table_values == mm) for mm in range(self.M) ]);
        return np.dot(LogBaseMeasure, num_tables);

    '''
    ==========================================================================================================================================
    Internal utiltiy functions for manipulating the graph
    ==========================================================================================================================================
    '''
    def _set_connection(self, node_from : int, node_to : int, layer : int) -> int:
        '''
        Changes which node connects to which in the hddCRP

        Args:
          node: int in range(self.N)
          layer: int in range(self.num_layers)

        Returns:
          (int) code number for what type of connection change occured.

        Raises:
          ValueError if proposed connection creates an invalid state

        Results:
          Connection/table values can change
        '''
        previous_node_to, previous_layer_to = self._get_next_node(node_from, layer);

        if(previous_node_to == node_to):
            return 0; # don't bother doing anything in this case: connection already set

        if(self._groupings_compact[node_from, layer] != self._groupings_compact[node_to, layer]):
            raise ValueError("invalid connection between groups")

        layer_to = layer;
        if(node_to == node_from and layer > 0):
            layer_to = layer-1;

        table_num_from = self._C_tables[node_from,layer];
        table_num_to   = self._C_tables[node_to,layer_to];

        Y_type_from = self._C_table_values[table_num_from]
        Y_type_to   = self._C_table_values[table_num_to];

        if((Y_type_to != Y_type_from) and (Y_type_to != hddCRPModel.UNKNOWN_OBSERVATION) and (Y_type_from != hddCRPModel.UNKNOWN_OBSERVATION)):
            raise ValueError("invalid connection between observed labels. " + "From: " + str(node_from) + " (" + str(Y_type_from) + "), to: " + str(node_to) + " (" + str(Y_type_to) + ")")

        is_cycle_from = self._C_is_cycle[node_from,layer];

        is_connecting_to_predecessor = self._get_if_node_leads_to_destination_within_layer(node_to, node_from, layer) # needs to do before changing _C_ptr

        # changes the ptr to the new node
        self._C_ptr[node_from, layer] = node_to; 
        # updates the predecessors
        if(previous_node_to != node_from):
            self._C_predecessors[layer][previous_node_to].remove(node_from)
        if(node_to != node_from):
            self._C_predecessors[layer][node_to].append(node_from)

        change_type = 0;
        if(table_num_from != table_num_to):
            if(is_cycle_from):
                change_type = 1;

                # combines tables completely
                table_num = min(table_num_from, table_num_to)
                nodes = np.isin(self._C_tables,[table_num_from, table_num_to]);
                nodes_from = self._C_tables == table_num_from;
                Y_type = Y_type_from
                if(Y_type == hddCRPModel.UNKNOWN_OBSERVATION):
                    Y_type = Y_type_to;

                # all in cycle from table_num_from are not in cycle
                self._C_is_cycle[nodes_from[:,layer],layer] = False

                # combine label counts
                self._C_num_labeled_in_table[table_num] = self._C_num_labeled_in_table[table_num_from] + self._C_num_labeled_in_table[table_num_to]
                # propogate _C_num_labeled_upstream through the old cycle to make valid
                pc = self._C_num_labeled_upstream[previous_node_to, previous_layer_to];
                node_c, layer_c = self._get_next_node(previous_node_to, previous_layer_to);
                while(True): # do while
                    self._C_num_labeled_upstream[node_c, layer_c] += pc;
                    pc = self._C_num_labeled_upstream[node_c, layer_c];
                    if(not self._C_is_cycle[node_c, layer_c]): # while
                        node_c, layer_c = self._get_next_node(node_c, layer_c);
                    else:
                        break;

                # re-number and label tables
                self._C_tables[nodes] = table_num;
                if(Y_type_to != Y_type_from):
                    self._C_y[nodes] = Y_type;
                    self._C_table_values[table_num] = Y_type;

                # delete old table
                table_num_for_deletion = max(table_num_from, table_num_to)
                self._C_table_values = np.delete(self._C_table_values, table_num_for_deletion);
                self._C_num_labeled_in_table = np.delete(self._C_num_labeled_in_table, table_num_for_deletion);
                self._C_table_counter -= 1;
                self._C_tables[self._C_tables >= table_num_for_deletion] -= 1
            else:
                change_type = 2;
                # keeps two tables

                # set label counts
                self._C_num_labeled_in_table[table_num_to]   += self._C_num_labeled_upstream[node_from, layer]
                self._C_num_labeled_in_table[table_num_from] -= self._C_num_labeled_upstream[node_from, layer]
                # reset _C_num_labeled_upstream
                node_c = previous_node_to;
                layer_c = previous_layer_to;
                while(True): # do while
                    self._C_num_labeled_upstream[node_c, layer_c] -= self._C_num_labeled_upstream[node_from, layer]
                    if(not self._C_is_cycle[node_c,layer_c]): # while
                        node_c, layer_c = self._get_next_node(node_c, layer_c);
                    else:
                        break

                node_c = node_to;
                layer_c = layer_to;
                while(True): # do while
                    self._C_num_labeled_upstream[node_c, layer_c] += self._C_num_labeled_upstream[node_from, layer]
                    if(not self._C_is_cycle[node_c,layer_c]): # while
                        node_c, layer_c = self._get_next_node(node_c, layer_c);
                    else:
                        break

                # If no labeled nodes in table_num_from, change to UNKNOWN
                if(self._C_num_labeled_in_table[table_num_from] == 0):
                    self._C_table_values[table_num_from] = hddCRPModel.UNKNOWN_OBSERVATION;
                    self._C_y[self._C_tables == table_num_from] = hddCRPModel.UNKNOWN_OBSERVATION;

                # If table_num_to had no labels, but does now, change to Y_type
                if(Y_type_to == hddCRPModel.UNKNOWN_OBSERVATION and self._C_num_labeled_in_table[table_num_to] > 0):
                    self._C_table_values[table_num_to] = Y_type_from;
                    self._C_y[self._C_tables == table_num_to] = Y_type_from
                
                # relables preceeding nodes from node_from to table_num_to
                self._propogate_label_backwards(node_from, layer, table_num_to)
                    
        else:
            if(is_cycle_from):
                change_type = 3; 
                # connecting within table case: not splitting, but changing the cycle

                # relabel cycle: node_from must still be in cycle
                self._C_is_cycle[self._C_tables == table_num_from] = False;
                node_c = node_from;
                layer_c = layer;
                cycle_nodes = []
                while(not self._C_is_cycle[node_c, layer_c]):
                    cycle_nodes.append(node_c)
                    self._C_is_cycle[node_c, layer_c] = True;
                    node_c, layer_c = self._get_next_node(node_c, layer_c);

                # propogate _C_num_labeled_upstream through the old cycle to make valid
                prev_labeled_nodes = self._C_num_labeled_upstream[previous_node_to, previous_layer_to]
                node_c, layer_c = self._get_next_node(previous_node_to, previous_layer_to);

                while(not self._C_is_cycle[node_c,layer_c]): 
                    self._C_num_labeled_upstream[node_c, layer_c] += prev_labeled_nodes
                    prev_labeled_nodes = self._C_num_labeled_upstream[node_c, layer_c] ;
                    node_c, layer_c = self._get_next_node(node_c, layer_c);

                for node_c in cycle_nodes:
                    self._reset_C_num_labeled_upstream_from_node(node_c, layer)

                # self._C_num_labeled_in_table[table_num] is constant
            elif(is_connecting_to_predecessor):
                change_type = 4;
                # SPLITS A TABLE  node belongs now to table_new, others belong to table_num
                table_num_new = self._C_table_counter;
                Y_new = hddCRPModel.UNKNOWN_OBSERVATION;
                num_labeled_in_new_table = self._C_num_labeled_upstream[node_from, layer];
                if(num_labeled_in_new_table > 0):
                    Y_new = Y_type_from;

                self._C_table_values = np.append(self._C_table_values,Y_new)
                self._C_table_counter += 1;

                # node is now in the cycle
                node_c = node_from;
                layer_c = layer; # layer shouldn't change here
                cycle_nodes = []
                while(not self._C_is_cycle[node_c, layer_c]):
                    cycle_nodes.append(node_c)
                    self._C_is_cycle[node_c, layer_c] = True;
                    node_c, layer_c = self._get_next_node(node_c, layer_c);

                # split label counts
                self._C_num_labeled_in_table = np.append(self._C_num_labeled_in_table, num_labeled_in_new_table)
                self._C_num_labeled_in_table[table_num_from] -= num_labeled_in_new_table

                # recompute self._C_num_labeled_upstream[node, layer] 
                node_c = previous_node_to
                layer_c = previous_layer_to
                while(True): # do while
                    self._C_num_labeled_upstream[node_c, layer_c] -= num_labeled_in_new_table
                    if(not self._C_is_cycle[node_c, layer_c]): # while
                        node_c, layer_c = self._get_next_node(node_c, layer_c);
                    else:
                        break;
                
                #  is there are quicker way to do this?
                for node_c in cycle_nodes:
                    self._propogate_label_backwards(node_c, layer, table_num_new);
                    self._reset_C_num_labeled_upstream_from_node(node_c, layer)
                # no need to change for anything else in the new table: all non-cycle nodes must still have valid count

                # if old table now has no nodes, unlabels it
                if(Y_type_from != hddCRPModel.UNKNOWN_OBSERVATION and self._C_num_labeled_in_table[table_num_from] == 0):
                    self._C_table_values[table_num_from] = hddCRPModel.UNKNOWN_OBSERVATION;
                    self._C_y[self._C_tables == table_num_from] = hddCRPModel.UNKNOWN_OBSERVATION
            else:
                change_type = 5;
                # reattach node to other part of table: doesn't change tables or cycle

                # recompute self._C_num_labeled_upstream[node, layer] in this case
                if(self._C_num_labeled_upstream[node_from, layer] > 0):
                    node_c = node_to
                    layer_c = layer
                    while(True): # do while
                        self._C_num_labeled_upstream[node_c, layer_c] += self._C_num_labeled_upstream[node_from, layer]
                        if(not self._C_is_cycle[node_c, layer_c]): # while
                            node_c, layer_c = self._get_next_node(node_c, layer_c);
                        else:
                            break
                    node_c = previous_node_to
                    layer_c = previous_layer_to
                    while(True): # do while
                        self._C_num_labeled_upstream[node_c, layer_c] -= self._C_num_labeled_upstream[node_from, layer]
                        if(not self._C_is_cycle[node_c, layer_c]): # while
                            node_c, layer_c = self._get_next_node(node_c, layer_c);
                        else:
                            break
                # self._C_num_labeled_in_table[table_num] is constant
        return change_type;

        
    def _get_next_node(self, node : int, layer : int) -> tuple[int,int]:
        '''
        Moves pointer to what it's connected to.

        Assumes:
              valid: _C_ptr: (int array size N x num_layers) 

        Args:
          node: int in range(self.N)
          layer: int in range(self.num_layers)
        Returns:
          (node, layer) : destination node
        '''
        node_c = self._C_ptr[node, layer];
        if(node_c == node):
            layer_c = max(0, layer-1);
        else:
            layer_c = layer;
        return (node_c, layer_c);


    def _get_preceeding_nodes_within_layer(self, node, layer) -> list:
        '''
        Finds all nodes with connections that point to the given node. Only considers nodes within layer!
        This is useful for a Gibbs sampling step.
        The node cannot be part of a cycle (this function would infinite loop otherwise).
        While it would be possible for a simple extension that just returns the entire table, that's not a case that should be calling this function and that's why the assert exists.

        Args:
          node: int in range(self.N)
          layer: int in range(self.num_layers)
        Returns:
          list of all nodes in the layer that end up pointing to node. Also includes the node!
        '''
        assert not self._C_is_cycle[node,layer], "node must not be in cycle!"

        prs = [node];

        ctr = 0;
        while(ctr < len(prs)):
            prs += self._C_predecessors[layer][prs[ctr]]
            ctr += 1;
        return prs

    def _get_if_node_leads_to_destination_within_layer(self, node_start, node_dest, layer):
        '''
        Finds if a node's connections leads to another node: only considers within layer

        Args:
          node: int in range(self.N)
          layer: int in range(self.num_layers)
        Returns:
          Whether or not node_start is upstream of node_dest
        '''
        if(self._C_tables[node_start,layer] != self._C_tables[node_dest,layer]):
            return False; # must be at same table
        if(self._C_is_cycle[node_dest, layer]):
            return True; # always true if the dest is in the ending cycle

        layer_start = layer;
        while(not self._C_is_cycle[node_start, layer_start] # if reached cycle, can't be connected
                and layer_start == layer):  # if escaped layer, not connected
            if(node_start == node_dest):
                return True; # connection found
            node_start, layer_start = self._get_next_node(node_start, layer_start)

        return False;
        
    def _reset_C_num_labeled_upstream_from_node(self, node_start, layer) -> None:
        '''
        Resets the values of _C_num_labeled_upstream from one specific node, traversing the graph forwards until the cycle of the table has been reached.

        Args:
          node_start: int in range(self.N) the start point
          layer: int in range(self.num_layers) which layer to begin in
        '''

        node_c = node_start;
        while(True): # do while
            self._C_num_labeled_upstream[node_c,layer] = int(self._C_y_0[node_c,layer] != hddCRPModel.UNKNOWN_OBSERVATION)
            if(layer < self.num_layers - 1 and self._C_ptr[node_c, layer+1] == node_c):
                self._C_num_labeled_upstream[node_c,layer] += self._C_num_labeled_upstream[node_c,layer+1]

            for gg in self._C_predecessors[layer][node_c]:
                if(not self._C_is_cycle[gg,layer]):
                    self._C_num_labeled_upstream[node_c,layer] += self._C_num_labeled_upstream[gg,layer]

            if(not self._C_is_cycle[node_c,layer]): # the while part
                node_c, layer = self._get_next_node(node_c, layer);
            else:
                break;

    def _propogate_label_backwards(self, node_from, layer, table_num_to) -> None:
        '''
        Resets the values of _C_tables and _C_y from one specific node, traversing the graph backwards until all preceding nodes have been reached.
        Changes the nodes to be at a new table. The _C_y values are taken from _C_table_values[table_num_to]

        Args:
          node_from: int in range(self.N) the start point
          layer: int in range(self.num_layers) which layer to begin in
          table_num_to: int in range(self._C_table_counter) the new table number for the nodes
        '''
        self._C_tables[node_from, layer] = table_num_to
        self._C_y[node_from, layer] = self._C_table_values[table_num_to]

        nns = self._C_predecessors[layer][node_from]
        for nn in nns:
            if(not self._C_is_cycle[nn,layer]):
                self._propogate_label_backwards(nn, layer, table_num_to);

        # any arrows from denser layers pointing to these nodes
        if(layer < self.num_layers - 1 and self._C_ptr[node_from, layer+1] == node_from):
            self._propogate_label_backwards(node_from, layer+1, table_num_to);

    def _compute_weights(self, weights : ArrayLike) -> np.ndarray:
        '''
        Computes the unnormalized probability of connections between each node.

        Args:
          weights: (length P) the parameters for the given weight function.
        Returns:
          The weights for each connection given the weight parameters
        '''
        F = np.apply_along_axis(lambda rr : self._weight_func(rr, weights), 2, self._D)
        np.fill_diagonal(F, 0);
        return F;


    '''
    ==========================================================================================================================================
    Utiltiy functions for debugging
    ==========================================================================================================================================
    '''
    def _validate_connections(self) -> None:
        if(~np.all(np.isin(self._C_ptr, range(self.N)))):
            print("CB_ptr \n" + str(self._CB_ptr.T))
            print("C_ptr \n" + str(self._C_ptr.T))
            raise ValueError("_C_ptr contains invalid pointer values")
        
        # check predecessors
        C_predecessors = [[[] for nn in range(self.N)] for ll in range(self.num_layers)] 
        for layer in range(self.num_layers-1, -1, -1):
            for node_from in range(self.N):
                node_to = self._C_ptr[node_from,layer];
                if(node_to != node_from): # no self connections
                    C_predecessors[layer][node_to].append(node_from);
        
        for layer in range(self.num_layers):
            for node in range(self.N):
                if(np.any(~np.isin(C_predecessors[layer][node], self._C_predecessors[layer][node]))):
                    print("CB_predecessors \n" + str(self._CB_predecessors[layer][node]))
                    print("C_predecessors \n" + str(self._C_predecessors[layer][node]))
                    print("recomputed predecessors \n" + str(C_predecessors[layer][node]))
                    raise ValueError("_C_predecessors is missing nodesin node " + str(node) + " layer " + str(layer))
                if(np.any(~np.isin(self._C_predecessors[layer][node], C_predecessors[layer][node]))):
                    print("CB_predecessors \n" + str(self._CB_predecessors[layer][node]))
                    print("C_predecessors \n" + str(self._C_predecessors[layer][node]))
                    print("recomputed predecessors \n" + str(C_predecessors[layer][node]))
                    raise ValueError("_C_predecessors has extra nodes in node " + str(node) + " layer " + str(layer) )

    def _validate_labels(self) -> None:
        if(self._C_table_values.size != self._C_table_counter):
            raise ValueError("_C_table_counter is not tracking properly")

        # check that all customers at a table have matching labels and that all custumers are seated
        found = np.zeros((self.N, self.num_layers), dtype=bool)
        for ii, label in enumerate(self._C_table_values):
            tt = self._C_tables == ii
            found[tt] = True;
            if(np.any(self._C_y[tt] != label)):
                print("CB_tables_values \n" + str(self._CB_table_values))
                print("CB_tables \n" + str(self._CB_tables.T))
                print("CB_y \n" + str(self._CB_y.T))
                print("C_table_values \n" + str(self._C_table_values))
                print("C_tables \n" + str(self._C_tables.T))
                print("C_y \n" + self._C_y.T)
                raise ValueError("Invalid table labeling: table number " + str(ii))
        if(np.any(~found)):
            print("CB_tables \n" + str(self._CB_tables.T))
            print("C_tables \n" + str(self._C_tables.T))
            raise ValueError("Not all customers seated!")

        # check to see that each connection points to same table
        for node in range(self.N):
            for layer in range(self.num_layers):
                table_from = self._C_tables[node, layer];

                node_to, layer_to = self._get_next_node(node, layer);
                table_to = self._C_tables[node_to, layer_to];
                if(table_from != table_to):
                    print("CB_tables_values \n" + str(self._CB_table_values))
                    print("CB_tables \n" + str(self._CB_tables.T))
                    print("CB_y \n" + str(self._CB_y.T))
                    print("CB_ptr \n" + str(self._CB_ptr.T))
                    print("C_table_values \n" + str(self._C_table_values))
                    print("C_tables \n" + str(self._C_tables.T))
                    print("C_y \n" + self._C_y.T)
                    print("C_ptr \n" + str(self._C_ptr.T))
                    raise ValueError("node " + str(node) + " in layer " + str(layer) + " points to a different table at node = " + str(node_to) + " layer " + str(layer_to) + "!")

    def _validate_cycles(self) -> None:
        C_is_cycle = np.zeros((self.N, self.num_layers),dtype=bool)
        for tt in range(self._C_table_counter):
            # visit count
            visits = np.zeros((self.N, self.num_layers),dtype=int);

            # get first node in table (doesn't matter where we start)
            node, layer = np.argwhere(self._C_tables == tt)[0]
            visits[node,layer] += 1;
            while(visits[node,layer] < 2):
                node, layer = self._get_next_node(node, layer);
                visits[node,layer] += 1;

            # now from current point, labels full cycle
            while(not C_is_cycle[node,layer]):
                C_is_cycle[node,layer] = True;
                node, layer = self._get_next_node(node, layer);
        if(np.any(C_is_cycle != self._C_is_cycle)):
            print("CB_is_cycle \n" + str(self._CB_is_cycle.T))
            print("C_is_cycle \n" + str(self._C_is_cycle.T))
            print("recomputed is_cycle \n" + str(C_is_cycle.T))
            print("CB_ptr \n" + str(self._CB_ptr.T))
            print("CB_tables \n" + str(self._CB_tables.T))
            print("CB_y \n" + str(self._CB_y.T))
            print("C_ptr \n" + str(self._C_ptr.T))
            print("C_tables \n" + str(self._C_tables.T))
            print("C_y \n" + self._C_y.T)
            raise ValueError("_C_is_cycle is not valid")
    
    def _validate_counts(self) -> None:
        if(self._C_table_values.size != self._C_num_labeled_in_table.size):
            raise ValueError("_C_num_labeled_in_table is not correct size!")

        for ii, label in enumerate(self._C_table_values):
            tt = self._C_tables == ii
            cnt = np.sum(self._C_y_0[tt] != hddCRPModel.UNKNOWN_OBSERVATION);
            if(cnt != self._C_num_labeled_in_table[ii]):
                print("recomputed num_labeled_in_table[ii] " + str(cnt))
                print("C_num_labeled_in_table[ii] " + str(self._C_num_labeled_in_table[ii]))
                raise ValueError("_C_num_labeled_in_table has invalid count: table number " + str(ii))

        # trace all labeled nodes
        labeled_coords = np.argwhere(self._C_y_0 != hddCRPModel.UNKNOWN_OBSERVATION);
        C_num_labeled_upstream = np.zeros((self.N, self.num_layers),dtype=int)
        C_num_labeled_upstream.fill(0)
        for pp in labeled_coords:
            node = pp[0];
            layer = pp[1];

            while(True): # do while
                C_num_labeled_upstream[node,layer] += 1;

                if(not self._C_is_cycle[node,layer]): # while
                    node, layer = self._get_next_node(node, layer);
                else:
                    break;
        if(np.any(C_num_labeled_upstream != self._C_num_labeled_upstream)):
            print("CB_num_labeled_upstream \n" + str(self._CB_num_labeled_upstream.T))
            print("C_num_labeled_upstream \n" + str(self._C_num_labeled_upstream.T))
            print("recomputed num_labeled_upstream \n" + str(C_num_labeled_upstream.T))
            raise ValueError("_C_num_labeled_upstream has invalid counts")

            
    def _validate_internals(self, code):
        try:
            self._validate_connections()
            self._validate_labels()
            self._validate_cycles()
            self._validate_counts()
        except ValueError as err:
            print("Error found with code: " + str(code))
            raise err





def Metropolis_Hastings_step_for_hddCRP_parameters(hddcrp : hddCRPModel, sigma2 : ArrayLike | float, log_prior_probability : Callable = None) -> tuple[hddCRPModel, float]:
    '''
    Takes a random-walk Metropolis-Hastings step for the hddCRP model parameters.
    Here, I assume all parameters are positive. So, the sampling is in LOG space.

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
    theta_current = np.concatenate([hddcrp.alpha.flatten(), hddcrp.weight_params.flatten()]);
    weight_idx = range(hddcrp.alpha.size, hddcrp.weight_params.size + hddcrp.alpha.size)
    alpha_idx = range(hddcrp.alpha.size)

    log_theta_current = np.log(theta_current)
    if(np.isscalar(sigma2)):
        eps = np.random.normal(scale=np.sqrt(sigma2),size=theta_current.shape)
    elif(np.size(sigma2) == np.size(theta_current)):
        eps = np.random.normal(scale=np.sqrt(np.array(sigma2).flatten()))
    elif(np.shape(sigma2) == (np.size(theta_current),)*2):
        eps = np.random.multivariate_normal(np.zeros_like(theta_current), sigma2)
    else:
        raise ValueError("invalid variance for random-walk Metropolis-Hastings step")

    log_theta_star = log_theta_current + eps;
    theta_star = np.exp(log_theta_star)

    log_p_Y_current = hddcrp.compute_log_likelihood()
    log_p_Y_star = hddcrp.compute_log_likelihood(weight_params=theta_star[weight_idx], alphas=theta_star[alpha_idx])

    log_P_theta_current = log_prior_probability(theta_current) + np.sum(log_theta_current)
    log_P_theta_star    = log_prior_probability(theta_star) + np.sum(log_theta_star)

    log_acceptance_probability = min(0.0, log_p_Y_star + log_P_theta_star - (log_p_Y_current + log_P_theta_current))

    aa = -np.random.exponential()

    if(aa > log_acceptance_probability):
        accepted = True
        hddcrp.alpha = theta_star[alpha_idx]
        hddcrp.weight_params = theta_star[weight_idx]
    else:
        accepted = False

    return (hddcrp, log_acceptance_probability, accepted)