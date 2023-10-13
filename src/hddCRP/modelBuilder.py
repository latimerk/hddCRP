
import numpy as np;
from numpy.typing import ArrayLike
import stan, stan.model, stan.fit
from hddCRP import stanModels
import arviz as az
import pandas as pd

class UNKNOWN_OBSERVATION:
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return True
        else:
            return False
        
class cdCRP_priorParams():
    def __init__(self, max_context_depth : int = 2, max_nback_depth : int = 1;):
        # defaults:
        self.alpha = {"shape" : 2.0,
                      "scale" : 5.0} # gamma prior

        self.timeconstant_within_session = {"shape" : 2.0,
                                            "scale" : 25.0} # gamma prior
        
        self.timeconstant_between_sessions = {"shape" : 2.0,
                                              "scale" : 2.0} # gamma prior
        
        self.context_similarity_0 = {"alpha" : 1.0,
                                     "beta"  : 1.0} # beta prior
        
        self.repeat_bias_0 = {"shape" : 20} # gamma prior fixed to mean 1

        self.context_depth = max_context_depth
        self.nback_depth   = max_nback_depth


    def __dict__(self):
        prior_params = {"prior_alpha_shape" : self.get_alpha_shape,
                        "prior_alpha_scale" : self.get_alpha_scale,
                        "prior_timeconstant_within_session_shape" : self.get_timeconstant_within_session_shape,
                        "prior_timeconstant_within_session_scale" : self.get_timeconstant_within_session_scale,
                        "prior_timeconstant_between_sessions_shape" : self.get_timeconstant_between_sessions_shape,
                        "prior_timeconstant_between_sessions_scale" : self.get_timeconstant_between_sessions_scale}
        for depth in range(1,self.context_depth+1):
            prior_params[f"prior_context_similarity_depth_{depth}_alpha"] = self.get_context_similarity_alpha(depth)
            prior_params[f"prior_context_similarity_depth_{depth}_beta"] = self.get_context_similarity_beta(depth)
        for depth in range(1,self.context_depth+1):
            prior_params[f"prior_repeat_bias_{depth}_back_shape"] = self.get_repeat_bias_shape(depth)
        return prior_params;

    @property
    def context_depth(self) -> int:
        return self._context_depth;
    @context_depth.setter
    def context_depth(self, cd : int) -> None:
        cd = int(cd)
        assert cd >= 0, "context_depth must be non-negative integer"
        self._context_depth = cd

    @property
    def nback_depth(self) -> int:
        return self._nback_depth;
    @nback_depth.setter
    def nback_depth(self, nd : int) -> None:
        nd = int(nd)
        assert nd > 0, "nback_depth must be non-negative integer"
        self._nback_depth = nd

    def set_alpha_shape(self, shape : float) -> None:
        assert shape > 0, "shape must be positive"
        self.alpha["shape"] = float(shape);
    def get_alpha_shape(self) -> float:
        return self.alpha["shape"];
    def set_alpha_scale(self, scale : float) -> None:
        assert scale > 0, "scale must be positive"
        self.alpha["scale"] = float(scale);
    def get_alpha_scale(self) -> float:
        return self.alpha["scale"];

    def set_timeconstant_within_session_shape(self, shape : float) -> None:
        assert shape > 0, "shape must be positive"
        self.timeconstant_within_session["shape"] = float(shape);
    def get_timeconstant_within_session_shape(self) -> float:
        return self.timeconstant_within_session["shape"];
    def set_timeconstant_within_session_scale(self, scale : float) -> None:
        assert scale > 0, "scale must be positive"
        self.timeconstant_within_session["scale"] = float(scale);
    def get_timeconstant_within_session_scale(self) -> float:
        return self.timeconstant_within_session["scale"];

    def set_timeconstant_between_sessions_shape(self, shape : float) -> None:
        assert shape > 0, "shape must be positive"
        self.timeconstant_between_sessions["shape"] = float(shape);
    def get_timeconstant_between_sessions_shape(self) -> float:
        return self.timeconstant_between_sessions["shape"];
    def set_timeconstant_between_sessions_scale(self, scale : float) -> None:
        assert scale > 0, "scale must be positive"
        self.timeconstant_between_sessions["scale"] = float(scale);
    def get_timeconstant_between_sessions_scale(self) -> float:
        return self.timeconstant_between_sessions["scale"];

    def set_context_similarity_alpha(self, alpha : float, depth : int | None) -> None:
        assert alpha > 0, "alpha must be positive"
        if((depth is None) or depth == 0):
            # default value for any unassigned back
            self.context_similarity_0["alpha"] = float(alpha);
        elif(int(depth) > 0):
            depth = int(depth);
            if not(depth in self.context_similarity_0.keys()):
                self.context_similarity_0[depth] = {"alpha" : float(alpha)};
            else:
                self.context_similarity_0[depth]["alpha"] = float(alpha)
        else:
            raise ValueError("nback must be non-negative integer or None")
    def get_context_similarity_alpha(self, depth : int | None) -> float:
        if((depth is None) or depth == 0):
            # default value for any unassigned back
            return self.context_similarity_0["alpha"];
        elif(int(depth) > 0):
            depth = int(depth);
            if not(depth in self.context_similarity_0.keys()):
                return self.context_similarity_0["alpha"];
            else:
                return self.context_similarity_0[depth]["alpha"]
        else:
            raise ValueError("depth must be non-negative integer or None")
        
    def set_context_similarity_beta(self, beta : float, depth : int | None) -> None:
        assert beta > 0, "beta must be positive"
        if((depth is None) or depth == 0):
            # default value for any unassigned back
            self.context_similarity_0["beta"] = float(beta);
        elif(int(depth) > 0):
            depth = int(depth);
            if not(depth in self.context_similarity_0.keys()):
                self.context_similarity_0[depth] = {"beta" : float(beta)};
            else:
                self.context_similarity_0[depth]["beta"] = float(beta)
        else:
            raise ValueError("nback must be non-negative integer or None")
    def get_context_similarity_beta(self, depth : int | None) -> float:
        if((depth is None) or depth == 0):
            # default value for any unassigned back
            return self.context_similarity_0["beta"];
        elif(int(depth) > 0):
            depth = int(depth);
            if not(depth in self.context_similarity_0.keys()):
                return self.context_similarity_0["beta"];
            else:
                return self.context_similarity_0[depth]["beta"]
        else:
            raise ValueError("depth must be non-negative integer or None")
        
    def set_repeat_bias_shape(self, shape : float, nback : int | None) -> None:
        assert shape > 0, "shape must be positive"
        if((nback is None) or nback == 0):
            # default value for any unassigned back
            self.repeat_bias_0["shape"] = float(shape);
        elif(int(nback) > 0):
            nback = int(nback);
            if not(nback in self.repeat_bias_0.keys()):
                self.repeat_bias_0[nback] = {"shape" : float(shape)};
            else:
                self.repeat_bias_0[nback]["shape"] = float(shape)
        else:
            raise ValueError("nback must be non-negative integer or None")
    def get_repeat_bias_shape(self, nback : int | None) -> float:
        if((nback is None) or nback == 0):
            # default value for any unassigned back
            return self.repeat_bias_0["shape"];
        elif(int(nback) > 0):
            nback = int(nback);
            if not(nback in self.repeat_bias_0.keys()):
                return self.repeat_bias_0["shape"];
            else:
                return self.repeat_bias_0[nback]["shape"]
        else:
            raise ValueError("nback must be non-negative integer or None")

class cdCRP():
    def __init__(self, sequences : list[ArrayLike] | ArrayLike, subject_labels : list[float|int] = None, session_times : list[float|int]=None, session_labels : ArrayLike =None, possible_observations : ArrayLike = None, nback_depth : int = 1, context_depth : int = 2):
        assert len(sequences) > 0, "sequences cannot be empty"
        if(np.isscalar(sequences[0])):
            sequences = [sequences];  # if is single session
        assert np.all([len(ss) > 0 for ss in sequences]), "individual sequences cannot be empty"

        sequences = [np.array(ss).flatten() for ss in sequences]

        self.sequences = sequences;

        if((session_labels is None) or len(session_labels == 0)):
            session_labels = ['A'] * self.num_sessions
        assert len(session_labels) == self.num_sessions, "if session_labels are given, must be of length equal to the number of sequences"
        self.session_labels = np.array(session_labels, dtype="object").flatten();
        
        if((subject_labels is None) or len(subject_labels == 0)):
            subject_labels = ['Sub_A'] * self.num_sessions
        assert len(subject_labels) == self.num_sessions, "if subject_labels are given, must be of length equal to the number of sequences"
        self.subject_labels = np.array(subject_labels, dtype="object").flatten();
    
        if(possible_observations is None): # all possible actions are observed
            possible_observations = np.unique(np.concatenate([np.unique(ss) for ss in sequences]))
        possible_observations.sort();
        assert np.all([np.all(np.isin(ss, possible_observations)) for ss in sequences]), "values in each sequence must be members of possible_observations"
        self.possible_observations = np.array(possible_observations, dtype="object").flatten();
    
        if(session_times is None):
            session_times = np.arange(0, self.num_sessions)

        assert np.size(session_times) == self.num_sessions, "session_times must have same number of sessions as sequences"
        self.session_times = np.array(session_times,dtype=float).flatten()

        self.priors = cdCRP_priorParams();
        self.nback_depth   = nback_depth;
        self.context_depth = context_depth

        if(self.is_population > 0):
            raise NotImplementedError("Population models cannot been created yet.")
        if(self.sessions_per_subject > 0):
            raise NotImplementedError("Multisession models cannot been created yet.")
        
        self.posterior = None;
        self.fit = None;

    @property
    def is_population(self) -> bool:
        return len(self.subjects) > 0;

    @property
    def num_sessions(self) -> int:
        return len(self.sequences);

    @property
    def session_lengths(self) -> np.ndarray[int]:
        return np.array([len(ss) for ss in self.sequences], dtype=int);

    @property
    def observations_per_subject(self) -> np.ndarray[int]:
        return np.array([np.sum(self.session_lengths[self.subject_labels == sub]) for sub in self.subjects])
    
    @property
    def sessions_per_subject(self) -> np.ndarray[int]:
        return np.array([np.sum(self.subject_labels == sub) for sub in self.subjects])

    @property
    def session_types(self) -> list[int]:
        return np.unique(self.session_labels);
    @property
    def num_session_types(self) -> list[int]:
        return len(self.session_types);

    @property
    def subjects(self) -> list[int]:
        return np.unique(self.subjects);

    @property
    def num_sessions(self) -> list[int]:
        return len(self.sequences);

    @property
    def total_observations(self) -> int:
        return np.sum(self.session_lengths)
    
    @property
    def M(self) -> int:
        return self.possible_observations.size
    
    @property
    def context_depth(self) -> int:
        return self._context_depth;
    @context_depth.setter
    def context_depth(self, cd : int) -> None:
        cd = int(cd)
        assert cd >= 0, "context_depth must be non-negative integer"
        self._context_depth = cd
        self.priors.context_depth = cd
    @property
    def nback_depth(self) -> int:
        return self._nback_depth;
    @nback_depth.setter
    def nback_depth(self, nd : int) -> None:
        nd = int(nd)
        assert nd > 0, "nback_depth must be non-negative integer"
        self._nback_depth = nd
        self.priors.nback_depth = nd
    
    
    def setup_compact_sequences(self, flatten=True) -> list[np.ndarray[int]]:
        v = np.arange(1,self.M+1).astype(int);
        seqs = [(np.array(ss)[:,np.newaxis] == self.possible_observations[np.newaxis,:]) @ v for ss in self.sequences]
        if(flatten):
            seqs = np.concatenate(seqs);
        return seqs;

    def flatten_sequences(self) -> list[np.ndarray[int]]:
        return np.concatenate([np.array(cc, dtype='object').flatten() for cc in self.sequences]);

    def setup_contexts(self, depth : int, flatten=True) -> np.ndarray:
        depth = int(depth)
        assert depth > 0, "depth must be positive integer"

        contexts = [];
        for seq in self.sequences:
            contexts += [[UNKNOWN_OBSERVATION()] * depth + [tuple(seq[ss-depth:ss]) for ss in range(len(seq))]];
        if(flatten):
            contexts = np.concatenate([np.array(cc, dtype='object').flatten() for cc in contexts]);

        return contexts;

    def setup_nback(self, depth : int, flatten=True) -> np.ndarray:
        contexts = [];
        for seq in self.sequences:
            contexts += [[UNKNOWN_OBSERVATION()] * depth + [seq[ss] for ss in range(len(seq))]];
        if(flatten):
            contexts = np.concatenate([np.array(cc, dtype='object').flatten() for cc in contexts]);
        return contexts;


    def __dict__(self):
        data = {"N" : self.total_observations, "M" : self.M, "T" : np.max(self.observations_per_subject),
                "Y" : self.setup_compact_sequences(flatten=True)}
        
        for depth in range(1, self.context_depth):
            contexts = self.setup_contexts(depth);
            match = contexts[:,np.ndarray] == contexts[np.ndarray,:]
            data[f"is_same_context_{depth}"] = np.tril(match,-1)

        base = self.flatten_sequences();
        for depth in range(1, self.nback_depth):
            nback = self.setup_nback(depth);
            match = base[:,np.ndarray] == nback[np.ndarray,:]
            data[f"is_same_context_{depth}"] = np.tril(match)

        data.update(dict(self.priors))
        return data;

    @property
    def data(self):
        return dict(self);

    @property
    def model(self):
        pop_model = self.is_population
        multisession_model = np.max(self.sessions_per_subject) > 1
        context_depth = self.context_depth
        repeat_depth = self.nback_depth

        model_str = ""
        if(pop_model):
            model_str += "population_"
        else:
            model_str += "individual_"
        if(multisession_model):
            model_str += "multisession_"
        else:
            model_str += "session_"
        model_str += f"context_{context_depth}" 
        model_str += f"repeat_{repeat_depth}" 


        if(not pop_model):
            num_session_interaction_types = self.num_session_types

            return stanModels.generate_stan_code_individual(num_session_interaction_types=num_session_interaction_types, context_depth=self.context_depth, nback_depth=self.nback_depth);

            # if(self.num_sessions == 1):
            #     if(repeat_depth == 0):
            #         if(context_depth == 0):
            #             return stanModels.model_individual_session_context_0_repeat_0
            #         elif(context_depth == 1):
            #             return stanModels.model_individual_session_context_1_repeat_0
            #         elif(context_depth == 2):
            #             return stanModels.model_individual_session_context_2_repeat_0
            #         else:
            #             raise NotImplementedError("No models with context depth > 2: " + model_str )
            #     elif(repeat_depth == 1):
            #         if(context_depth == 0):
            #             return stanModels.model_individual_session_context_0_repeat_1
            #         elif(context_depth == 1):
            #             return stanModels.model_individual_session_context_1_repeat_1
            #         elif(context_depth == 2):
            #             return stanModels.model_individual_session_context_2_repeat_1
            #         else:
            #             raise NotImplementedError("No models with context depth > 2: " + model_str )
            #     else:
            #         raise NotImplementedError("No models with repeat depth > 1: " + model_str )
            # else:
            #     if(self.num_session_types == 1):
            #         raise NotImplementedError("No multisession models with different session labels yet: " + model_str )
        else:
            raise NotImplementedError("No population models yet: " + model_str )
        

    def build(self, random_seed : int) -> stan.model.Model:
        self.posterior = stan.build(self.model, data=self.data, random_seed=int(random_seed))
        self.posterior.random_seed

    def fit(self, num_chains : int = 4, num_samples : int = 1000, random_seed : int = None) -> stan.fit.Fit:
        if(self.posterior is None):
            if(not (random_seed is None)):
                self.build(random_seed=random_seed);
            else:
                raise ValueError("Model not built yet!")
        
        num_chains = int(num_chains)
        num_samples = int(num_samples)

        assert num_chains  > 0, "must have positive number of chains for sampling"
        assert num_samples > 0, "must have positive number of samples per chain"

        self.fit = self.posterior.sample(num_chains=num_chains, num_samples=num_samples)
        return self.fit;

    def fit_summary(self) -> pd.DataFrame:
        if(self.fit is None):
            raise ValueError("Model not fit yet!")
        
        return az.summary(self.fit)