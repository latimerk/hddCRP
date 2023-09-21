import numpy as np;

# for typing and validating arguments
from numpy.typing import ArrayLike
from typing import Callable
from inspect import signature

# special functions
from scipy.stats import gamma
from scipy.special import logsumexp

from itertools import product

import warnings

class sequentialhddCRPModel():
    '''
    Holds info for a hierarchical distance dependent Chinese restaurant process (hddCRP) model where we have exact, discrete observations (N observations labeled 1 to M).
    This class helps with Gibbs sampling the process's latent variables given the data, as well as computing log-likelihoods (for Metropolis-Hastings steps over parameters) and predicitve inference.
    '''
    UNKNOWN_OBSERVATION = -1;

    def __init__(self, Y : ArrayLike, groupings : None | ArrayLike,
                       alpha : float | ArrayLike,
                       Y_values : ArrayLike = None, BaseMeasure : ArrayLike | None = None,
                       weight_params : float | ArrayLike = 1, weight_param_labels : list[str] = None,
                       weight_func : Callable = lambda x, y : np.exp(-np.abs(x)/y), rng : np.random.Generator = None) -> None:
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
          Y_values: (length M) The possible values Y can take (redundant if all values observed - this is just in case). If None, it contains the unique
                    elements of Y.
          BaseMeasure: (length M or empty) The base measure of observations. If None (or scalar), then assumes uniform distribution over Y_values.
          weight_params: (length P) The parameters for the distance-to-weight function
          weight_func: (length N-1 array - distances (range(1,N)), length P array - params) -> non-negative scalar
                       Function for taking a single distance between two observations into the weight for connection from one node to the next in the hddCRP.
          weight_param_labels: (length P) List of names for each of the weight_params values
          rng : random number generator to use - is np.random.default_rng() by default
        
        """
        if(rng is None):
            rng = np.random.default_rng()
        self._rng = rng;

        # sets _Y to a vector of ints. All negative numbers and nans are changed to a single index of UNKNOWN_OBSERVATION
        # makes the N property valid
        self._Y = np.array(Y).flatten()


        if(not Y_values is None):
            # If Y_values is given, sets it up as a vector similar to Y.
            self._Y_values = np.array(Y_values).flatten()
        elif(not Y is None):
            self._Y_values = np.unique(self._Y)

        # removes UNKNOWN_OBSERVATION from self._Y_values
        self._Y_values = np.unique(self._Y_values[~np.isin(self._Y_values,[np.nan, None])]);

        #checks to see if all values of Y are in Y_values. Note that Y_values does not need to contain UNKNOWN_OBSERVATION/nan
        Y_is_not_nan = np.equal(self._Y, self._Y)#~np.isnan(self._Y);
        assert np.all(~Y_is_not_nan), "Currently does not support NaNs in Y"

        # records values indicies of each type of Y
        self._Y_unknowns = np.where(~Y_is_not_nan)[0];
        self._Y_indicies = [np.where(self._Y == xx)[0] for xx in self._Y_values];
        self._Y_compact = np.zeros((self.N)) # makes sure numbers for Y labels are in an easier to use, compact form (if obs labels are "A","B","Z" becomes 0,1,2)
        self._Y_compact.fill(np.nan)
        for ynum, yys in enumerate(self._Y_indicies):
            self._Y_compact[yys] = ynum;
        self._Y_compact = self._Y_compact.astype(int)

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

        self._weights = np.zeros((self.N,self.N)); # the actual weights to be computed from the distances given the parameters and function

        # sets up the weighting function
        if(not weight_func is None or complete_weight_func is None):
            assert isinstance(weight_func, Callable), "weight_func must be a function"
            sig = signature(weight_func)
            assert len(sig.parameters) == 2, "weight_func must take two parameters"
            self._weight_func = weight_func;
            assert self._weight_func(self._D[0,0,:], weight_params) >= 0, "weight function may not produce valid results" # tries running the weight function to make sure it works

        if(complete_weight_func is None):
            complete_weight_func = lambda D, weights : np.apply_along_axis(lambda rr : self._weight_func(rr, weights), 2, D)
        self._complete_weight_func = complete_weight_func;

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

        # sets up group index information
        self._previous_in_group_distances = []; # layer x observation x num in group
        self._previous_in_group_distances_same_observation = []; # layer x observation x num in group with same observation
        self._previous_in_group_distances_matrix = np.zeros((self.N, self.N+1, self.num_layers), dtype=bool)
        self._previous_in_group_distances_same_observation_matrix = np.zeros((self.N+1, self.N+1, self.num_layers), dtype=bool)
        for layer in range(self.num_layers):
            prev_in_group_c = [];
            prev_in_group_same_obs_c = [];
            #for each observation
            for nn in range(self.N):
                group_match = self._groupings_compact[0:nn, layer] == self._groupings_compact[nn, layer];
                obs_match = self._Y_compact[0:nn] == self._Y_compact[nn];

                prev_in_group_c += [nn - np.where(group_match)[0]]
                prev_in_group_same_obs_c += [nn - np.where(group_match and obs_match)[0]]

                self._previous_in_group_distances_matrix[nn, nn - np.where(group_match)[0], layer] = 1;
                self._previous_in_group_distances_same_observation_matrix[nn, nn - np.where(group_match and obs_match)[0], layer] = 1;
            
            self._previous_in_group_distances += prev_in_group_c
            self._previous_in_group_distances_same_observation += prev_in_group_same_obs_c
                
                    

    
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
        self._F = self._weight_func(np.arange(0,self.N+1),self._weight_params)
        assert np.all(self._F >= 0), "invalid connection weights found! check the weight/distance function"

        self._prob_group = self._compute_sum_prob_group()
        self._prob_group_same_obs = self._compute_sum_prob_group_same_observation()

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
    log likelihood computations
    ==========================================================================================================================================
    '''

    def _compute_sum_prob_group(self, F : ArrayLike = None, previous_in_group_distances = None, predictive : bool = False  ) -> ArrayLike:
        if(F is None):
            F = self._F
        if(previous_in_group_distances is None):
            if(predictive):
                #previous_in_group_distances = self._predictive_transition_probability_setup["test_previous_in_group_distances"]
                previous_in_group_distances_mat = self._predictive_transition_probability_setup["test_previous_in_group_distances_matrix"]
            else:
                # previous_in_group_distances = self._previous_in_group_distances;
                previous_in_group_distances_mat = self._previous_in_group_distances_matrix;
        else:
            previous_in_group_distances_mat = previous_in_group_distances;
        n_obs = previous_in_group_distances_mat.shape[0]
        F = np.array(F).flatten()
        PCs = np.zeros((n_obs, self.num_layers));
        
        # for layer, llg in enumerate(previous_in_group_distances):
        #     for obs, gg in enumerate(llg):
        #         PCs[obs, layer] = np.sum(F[gg])
        for layer in range(self.num_layers):
            PCs[:, layer] = previous_in_group_distances_mat[:,:,layer] @ F
        return PCs
    
    def _compute_sum_prob_group_same_observation(self, F : ArrayLike = None, previous_in_group_distances_same_observation = None, predictive : bool = False ) -> ArrayLike:
        if(F is None):
            F = self._F
        if(previous_in_group_distances_same_observation is None):
            if(predictive):
                #previous_in_group_distances_same_observation = self._predictive_transition_probability_setup["test_previous_in_group_distances_per_obs"]
                previous_in_group_distances_same_observation_mat = self._predictive_transition_probability_setup["test_previous_in_group_distances_per_obs_matrix"]
            else:
                #previous_in_group_distances_same_observation = self._previous_in_group_distances_same_observation;
                previous_in_group_distances_same_observation_mat = self._previous_in_group_distances_same_observation_matrix;
        else:
            previous_in_group_distances_same_observation_mat = previous_in_group_distances_same_observation;
        n_obs = previous_in_group_distances_same_observation_mat.shape[0]
        F = np.array(F).flatten()
        if(predictive):
            PCs = np.zeros((n_obs, self.num_layers, self.M));
        else:
            PCs = np.zeros((n_obs, self.num_layers));
        for layer, lly in enumerate(self._previous_in_group_distances_same_observation):
            if(predictive):
                # for obs, lly_mm in enumerate(lly):
                #     for mm, yy in enumerate(lly_mm):
                #         PCs[obs, layer, mm] = np.sum(F[yy])
                for mm in range(self.M):
                    for layer in range(self.num_layers):
                        PCs[:, layer, mm] = previous_in_group_distances_same_observation_mat[:,:,layer, mm] @ F
            else:
                # for obs, yy in enumerate(lly):
                #     PCs[obs, layer] = np.sum(F[yy])
                for layer in range(self.num_layers):
                    PCs[:, layer] = previous_in_group_distances_same_observation_mat[:,:,layer] @ F
        return PCs
    
    def _compute_log_p_stay_given_level_and_obs_log_p_level(self, alphas : ArrayLike = None, prob_group : ArrayLike = None, prob_group_same_obs : ArrayLike = None):
        if(alphas is None):
            alphas = self.alpha
        elif(np.isscalar(alphas)):
            alphas = np.ones((self.num_layers))*alphas
        alphas = np.array(alphas).flatten()
        assert np.size(alphas) == self.num_layers, "alphas must contain num_layers values (or scalar)"
        
        if(prob_group is None):
            prob_group = self._prob_group
        assert np.size(prob_group) == (self.N, self.num_layers) , "prob_group must be size N x num_layers"
        if(prob_group_same_obs is None):
            prob_group_same_obs = self._prob_group_same_obs
        assert np.size(prob_group_same_obs) == (self.N, self.num_layers) , "prob_group_same_obs must be size N x num_layers"

        log_P_stay_given_level_and_obs = np.zeros((self.N, self.num_layers))
        log_P_stay_given_level = np.zeros((self.N, self.num_layers))
        log_P_jump_given_level = np.zeros((self.N, self.num_layers))
        log_P_jump_given_level[:,0] = np.nan

        log_P_stay_given_level_and_obs[:,0] = np.log(self.BaseMeasure(self._Y_compact) * alphas[0] + prob_group_same_obs[:,0]) - np.log(alphas[0] + prob_group[:,0]);
        for layer in range(1,self.num_layers):
            log_n = np.log(alphas[layer] + prob_group[:,layer])
            log_P_stay_given_level_and_obs[:,layer] = np.log(prob_group_same_obs[:,layer]) - log_n;
            log_P_stay_given_level[:,layer] = np.log(prob_group[:,layer]) - log_n;
            log_P_jump_given_level[:,layer] = np.log(alphas[layer]) - log_n;
    
        log_P_level = np.zeros((self.N, self.num_layers))
        log_P_level[:,-1] = log_P_stay_given_level[:,-1]

        view=np.flip(log_P_jump_given_level,1)
        np.cumsum(log_P_jump_given_level,axis=1,out=view)
        for layer in range(1,self.num_layers-1):
            log_P_level[:,layer] = log_P_stay_given_level[:,layer] + log_P_jump_given_level[:,layer+1:]

        return log_P_level, log_P_stay_given_level_and_obs
    
    def compute_log_likelihood(self, alphas : ArrayLike = None, weight_params : ArrayLike = None):
        if(alphas is None):
            alphas = self.alpha
        elif(np.isscalar(alphas)):
            alphas = np.ones((self.num_layers))*alphas
        alphas = np.array(alphas).flatten()
        assert np.size(alphas) == self.num_layers, "alphas must contain num_layers values (or scalar)"

        if(weight_params is None):
            prob_group = self._prob_group
            prob_group_same_obs = self._prob_group_same_obs
        else:
            if(np.isscalar(weight_params)):
                weight_params = np.ones_like(self._weight_params)*weight_params
            weight_params = np.array(weight_params)
            F = self._weight_func(np.arange(0,self.N+1),weight_params)
            prob_group = self._compute_sum_prob_group(F)
            prob_group_same_obs = self._compute_sum_prob_group_same_observation(F)
    
        log_P_level, log_P_stay_given_level_and_obs = self._compute_log_p_stay_given_level_and_obs_log_p_level(alphas=alphas, prob_group=prob_group, prob_group_same_obs=prob_group_same_obs)

        log_like_per = logsumexp(log_P_level + log_P_stay_given_level_and_obs,axis=1)
        log_like = log_like_per.sum()

        return log_like
        
    def setup_transition_probability_computations(self, observation_indices=None):
        prefixes = [str(xx) + '-' for xx in self._Y_values]
        combinations = list(product(prefixes,repeat=self.num_layers-1))

        groups_at_each_level = [[]] * self.num_layers
        groups_at_each_level[0] = ['' for xx in combinations]
        for layer in range(1,self.num_layers):
            groups_at_each_level[layer] = [''.join(xx[-layer:]) for xx in combinations]

        groups_numbers_each_level = [[]] * self.num_layers
        for layer in range(0,self.num_layers):
            groups_numbers_each_level[layer] = [int(self._groupings_compact[self._groupings[:,layer] == xx,layer][0]) if np.any(np.isin(xx,self._groupings[:,layer])) else self.num_groups[layer] for xx in groups_at_each_level[layer]]

        contexts = [''.join(xx) for xx in combinations]

        # get weights for specified observations
        if(observation_indices is None):
            observation_indices = np.array([self.N]);
        
        # sets up group index information
        test_previous_in_group_distances = []; # layer x observation x num in group
        test_previous_in_group_distances_per_obs = [];  # layer x observation x Y type x num in group 
        test_previous_in_group_distances_matrix = np.zeros((len(contexts), self.N+1,self.num_layers) , dtype=bool)
        test_previous_in_group_distances_per_obs_matrix = np.zeros((len(contexts), self.N+1,self.num_layers,self.M),dtype=bool )
        for layer in range(self.num_layers):
            prev_in_group_c = [];
            prev_in_group_obs_c = [];
            #for each observation
            for nn in observation_indices:
                group_match = self._groupings_compact[:, layer] == self._groupings_compact[:, layer];
                nns = np.where(group_match)[0];
                nns = nns[nns < nn]
                prev_in_group_c += [nn - nns]
                test_previous_in_group_distances_matrix[nn, nn - nns, layer] = 1

                obs_c = [];
                for mm in range(self.M):
                    nns2 = nns[self._Y_compact[nns] == mm];
                    obs_c += [nn - nns2]
                    test_previous_in_group_distances_per_obs_matrix[nn, nn - nns2, layer, mm] = 1
                prev_in_group_obs_c += obs_c;
            
            test_previous_in_group_distances += prev_in_group_c
            test_previous_in_group_distances_per_obs += prev_in_group_obs_c

        self._predictive_transition_probability_setup = {"contexts" : contexts,
                                                         "observation_indices" : observation_indices,
                                                         "test_previous_in_group_distances" : test_previous_in_group_distances,
                                                         "test_previous_in_group_distances_per_obs" : test_previous_in_group_distances_per_obs,
                                                         "test_previous_in_group_distances_matrix" : test_previous_in_group_distances_matrix,
                                                         "test_previous_in_group_distances_per_obs_matrix" : test_previous_in_group_distances_per_obs_matrix}

    def compute_preditive_transition_probabilities(self, alphas : ArrayLike = None, weight_params : ArrayLike = None):
        if(alphas is None):
            alphas = self.alpha
        elif(np.isscalar(alphas)):
            alphas = np.ones((self.num_layers))*alphas
        alphas = np.array(alphas).flatten()
        assert np.size(alphas) == self.num_layers, "alphas must contain num_layers values (or scalar)"

        if(weight_params is None):
            F = self._F
        else:
            if(np.isscalar(weight_params)):
                weight_params = np.ones_like(self._weight_params)*weight_params
            weight_params = np.array(weight_params)
            F = self._weight_func(np.arange(0,self.N+1),weight_params)
        prob_group = self._compute_sum_prob_group(F, predictive=True)
        prob_group_same_obs = self._compute_sum_prob_group_same_observation(F, predictive=True)
        
        num_contexts = len(self._predictive_transition_probability_setup["contexts"])
        log_P_jumped = np.zeros((num_contexts))

        log_P_obs = np.zeros((num_contexts, self.M, self.num_layers))

        for layer in range(self.num_layers,0,-1):
            # n = (alphas[layer] + prob_group[:,layer]);
            log_n = (alphas[layer] + prob_group[:,layer]);
            #P_obs += P_jumped * (prob_group_same_obs[:,layer,:].squeeze(axis=1))/n[:,np.newaxis]
            log_P_obs[:,:,layer] = log_P_jumped  + np.log(prob_group_same_obs[:,layer,:].squeeze(axis=1)) - log_n[:,np.newaxis]
            # P_jumped *= alphas[layer]/n;
            log_P_jumped += np.log(alphas[layer]) - log_n;
        layer = 0;
        # n = (alphas[layer] + prob_group[:,layer]);
        # P_obs += P_jumped * (prob_group_same_obs[:,layer,:].squeeze(axis=1) + alphas[layer]*self.BaseMeasure[np.newaxis,:])/n[:,np.newaxis]
        # P_obs = P_obs / np.sum(P_obs,axis=1)
        log_n = np.log_(alphas[layer] + prob_group[:,layer]);
        log_P_obs[:,:,layer] = log_P_jumped  + np.log(prob_group_same_obs[:,layer,:].squeeze(axis=1) + alphas[layer]*self.BaseMeasure[np.newaxis,:]) - log_n[:,np.newaxis]
        log_P_obs = logsumexp(log_P_obs,axis=2)
        log_P_obs -= logsumexp(log_P_obs,axis=1)
        P_obs = np.exp(log_P_obs )
        return P_obs
    
    def compute_preditive_transition_probabilities_markov(self, layer : int, alpha : float = 0):
        layer = int(layer)
        assert np.isscalar(alpha) and alpha >= 0, "alpha be non-negative scalar"
        assert layer >= 0 and layer < self.num_layers, "invalid layer"

        F = np.ones_like(self._F)
        #prob_group = self._compute_sum_prob_group(F, predictive=True)
        prob_group_same_obs = self._compute_sum_prob_group_same_observation(F, predictive=True)
        
        #num_contexts = len(self._predictive_transition_probability_setup["contexts"])

        P_obs = (prob_group_same_obs[:,layer,:].squeeze(axis=1) +  alpha*self.BaseMeasure[np.newaxis,:])
        
        P_obs = P_obs / np.sum(P_obs,axis=1)
        return P_obs