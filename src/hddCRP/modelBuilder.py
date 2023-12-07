
import numpy as np;
from numpy.typing import ArrayLike
import stan, stan.model, stan.fit
from hddCRP import stanModels
import arviz as az
import pandas as pd

import asyncio
import stan.common

from scipy.optimize import minimize


class UNKNOWN_OBSERVATION:
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return True
        else:
            return False
        
class cdCRP_priorParams():
    def __init__(self, max_context_depth : int = 2, max_same_nback_depth : int = 1):
        # defaults:
        self.alpha = {"shape" : 2.0,
                      "scale" : 2.0} # gamma prior

        self.timeconstant_within_session = {"shape" : 2.0,
                                            "scale" : 20.0} # gamma prior
        
        self.timeconstant_between_sessions = {"shape" : 2.0,
                                              "scale" : 2.0} # gamma prior
        
        self.context_similarity_0 = {"alpha" : 1.0,
                                     "beta"  : 1.0} # beta prior
        
        self.repeat_bias_0 = {"shape" : 20} # gamma prior fixed to mean 1

        self.context_depth = max_context_depth
        self.same_nback_depth   = max_same_nback_depth


    def to_dict(self) -> dict:
        prior_params = {"prior_alpha_shape" : self.get_alpha_shape(),
                        "prior_alpha_scale" : self.get_alpha_scale(),
                        "prior_timeconstant_within_session_shape" : self.get_timeconstant_within_session_shape(),
                        "prior_timeconstant_within_session_scale" : self.get_timeconstant_within_session_scale(),
                        "prior_timeconstant_between_sessions_shape" : self.get_timeconstant_between_sessions_shape(),
                        "prior_timeconstant_between_sessions_scale" : self.get_timeconstant_between_sessions_scale()}
        for depth in range(1,self.context_depth+1):
            prior_params[f"prior_context_similarity_depth_{depth}_alpha"] = self.get_context_similarity_alpha(depth)
            prior_params[f"prior_context_similarity_depth_{depth}_beta"] = self.get_context_similarity_beta(depth)
        for depth in range(1,self.same_nback_depth+1):
            prior_params[f"prior_repeat_bias_{depth}_back_shape"] = self.get_repeat_bias_shape(depth)
        #yield from prior_params.items();
        return prior_params

    @property
    def context_depth(self) -> int:
        return self._context_depth;
    @context_depth.setter
    def context_depth(self, cd : int) -> None:
        cd = int(cd)
        assert cd >= 0, "context_depth must be non-negative integer"
        self._context_depth = cd

    @property
    def same_nback_depth(self) -> int:
        return self._same_nback_depth;
    @same_nback_depth.setter
    def same_nback_depth(self, nd : int) -> None:
        nd = int(nd)
        assert nd >= 0, "same_nback_depth must be non-negative integer"
        self._same_nback_depth = nd

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
            raise ValueError("same_nback must be non-negative integer or None")
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
            raise ValueError("same_nback must be non-negative integer or None")
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
        
    def set_repeat_bias_shape(self, shape : float, same_nback : int | None) -> None:
        assert shape > 0, "shape must be positive"
        if((same_nback is None) or same_nback == 0):
            # default value for any unassigned back
            self.repeat_bias_0["shape"] = float(shape);
        elif(int(same_nback) > 0):
            same_nback = int(same_nback);
            if not(same_nback in self.repeat_bias_0.keys()):
                self.repeat_bias_0[same_nback] = {"shape" : float(shape)};
            else:
                self.repeat_bias_0[same_nback]["shape"] = float(shape)
        else:
            raise ValueError("same_nback must be non-negative integer or None")
    def get_repeat_bias_shape(self, same_nback : int | None) -> float:
        if((same_nback is None) or same_nback == 0):
            # default value for any unassigned back
            return self.repeat_bias_0["shape"];
        elif(int(same_nback) > 0):
            same_nback = int(same_nback);
            if not(same_nback in self.repeat_bias_0.keys()):
                return self.repeat_bias_0["shape"];
            else:
                return self.repeat_bias_0[same_nback]["shape"]
        else:
            raise ValueError("same_nback must be non-negative integer or None")

class cdCRP():
    def __init__(self, sequences : list[ArrayLike] | ArrayLike,
                 subject_labels : list[float|int] = None,
                 population_labels : list[float|int] = None,
                 session_times : list[float|int]=None,
                 session_labels : ArrayLike =None,
                 possible_observations : ArrayLike = None,
                 same_nback_depth : int = 1, context_depth : int = 2):
        assert len(sequences) > 0, "sequences cannot be empty"
        if(np.isscalar(sequences[0])):
            sequences = [sequences];  # if is single session
        assert np.all([len(ss) > 0 for ss in sequences]), "individual sequences cannot be empty"

        sequences = [np.array(ss).flatten() for ss in sequences]

        self.sequences = sequences;

        if((session_labels is None)):
            session_labels = ['A'] * self.num_sessions
        if(np.isscalar(session_labels)):
            session_labels = [session_labels] * self.num_sessions
        assert len(session_labels) == self.num_sessions, "if session_labels are given, must be of length equal to the number of sequences"
        self.session_labels = np.array(session_labels, dtype="object").flatten();
        
        if((subject_labels is None)):
            subject_labels = ['Sub_A'] * self.num_sessions
        if(np.isscalar(subject_labels)):
            subject_labels = [subject_labels] * self.num_sessions
        assert len(subject_labels) == self.num_sessions, "if subject_labels are given, must be of length equal to the number of sequences"
        self.subject_labels = np.array(subject_labels, dtype="object").flatten();
    
        if((population_labels is None)):
            population_labels = self.subject_labels
        if(np.isscalar(population_labels)):
            population_labels = [population_labels] * self.num_sessions
        assert len(population_labels) == self.num_sessions, "if population_labels are given, must be of length equal to the number of sequences"
        self.population_labels = np.array(population_labels, dtype="object").flatten();

        for sub in self.subjects:
            assert np.unique(self.population_labels[self.subject_labels == sub]).size == 1, "subject cannot be placed in multiple populations in different sessions"

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
        self.same_nback_depth   = same_nback_depth;
        self.context_depth = context_depth

        
        self.distinct_session_within_session_timeconstants = True

        self.posterior = None;
        self.fit = None;
    
        self.session_interactions_enabled = True
        self.within_session_decay_enabled = True

        self.population_shared_within_session_timescale  = True;
        self.population_shared_between_session_timescale = True;
        self.population_shared_alpha      = True;
        self.population_shared_context    = [True] * self.context_depth;
        self.population_shared_same_nback = [True] * self.same_nback_depth;


    @property
    def fixed_alpha(self) -> float | None:
        if(not hasattr(self, "_fixed_alpha")):
            self._fixed_alpha = None;
        return self._fixed_alpha
    @fixed_alpha.setter
    def fixed_alpha(self, alpha : None | float) -> None:
        if(not (alpha is None)):
            assert np.isscalar(alpha), "fixed_alpha must be None or scalar"
            alpha = float(alpha)
            if(alpha <= 0):
                alpha = None;
        self._fixed_alpha = alpha

    
    @property
    def is_population_partial(self) -> bool:
        return self.is_population and (not (np.all(self.population_shared_context) and 
                                       np.all(self.population_shared_same_nback) and
                                       self.population_shared_alpha and
                                       self.population_shared_between_session_timescale and
                                       self.population_shared_within_session_timescale ));

    @property
    def is_population(self) -> bool:
        return self.num_subjects > 1;

    @property
    def is_multipopulation(self) -> bool:
        return self.num_populations > 1;


    @property
    def num_sessions(self) -> int:
        return len(self.sequences);

    @property
    def session_lengths(self) -> np.ndarray:
        return np.array([len(ss) for ss in self.sequences], dtype=int);

    @property
    def observations_per_subject(self) -> np.ndarray:
        return np.array([np.sum(self.session_lengths[self.subject_labels == sub]) for sub in self.subjects])
    
    @property
    def sessions_per_subject(self) -> np.ndarray:
        return np.array([np.sum(self.subject_labels == sub) for sub in self.subjects])

    @property
    def session_types(self) -> list[int]:
        return np.unique(self.session_labels);
    @property
    def num_session_types(self) -> list[int]:
        return len(self.session_types);

    @property
    def populations(self) -> list[int]:
        return np.unique(self.population_labels);
    @property
    def num_populations(self) -> int:
        return len(self.populations);

    @property
    def subjects(self) -> list[int]:
        return np.unique(self.subject_labels);
    @property
    def num_subjects(self) -> int:
        return len(self.subjects);

    @property
    def num_sessions(self) -> list[int]:
        return len(self.sequences);

    @property
    def total_observations(self) -> int:
        return np.sum(self.session_lengths)
    
    
    @property
    def distinct_session_within_session_timeconstants(self) -> bool:
        if(not hasattr(self, "_distinct_session_within_session_timeconstants")):
            self._distinct_session_within_session_timeconstants = True;
        return self._distinct_session_within_session_timeconstants
    @distinct_session_within_session_timeconstants.setter
    def distinct_session_within_session_timeconstants(self, distinct_time_constants : bool) -> None:
        self._distinct_session_within_session_timeconstants = distinct_time_constants
    
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
    def same_nback_depth(self) -> int:
        return self._same_nback_depth;
    @same_nback_depth.setter
    def same_nback_depth(self, nd : int) -> None:
        nd = int(nd)
        assert nd >= 0, "same_nback_depth must be non-negative integer"
        self._same_nback_depth = nd
        self.priors.same_nback_depth = nd
    
    
    def setup_compact_sequences(self, flatten : bool = True) -> list[np.ndarray]:
        v = np.arange(1,self.M+1).astype(int);
        seqs = [(np.array(ss)[:,np.newaxis] == self.possible_observations[np.newaxis,:]) @ v for ss in self.sequences]
        if(flatten):
            if(self.is_population):
                seqs = np.concatenate(self._stack_arrays_by_subject(seqs))
            else:
                seqs = np.concatenate(seqs);
        return seqs;

    def flatten_sequences(self) -> list[np.ndarray]:
        if(self.is_population):
            return self._stack_arrays_by_subject([np.array(cc, dtype='object').flatten() for cc in self.sequences]);
        else:
            return np.concatenate([np.array(cc, dtype='object').flatten() for cc in self.sequences]);

    def setup_contexts(self, depth : int, flatten : bool = True) -> np.ndarray:
        depth = int(depth)
        assert depth > 0, "depth must be positive integer"

        contexts = [];
        for seq in self.sequences:
            seq_alt = np.concatenate([np.array([UNKNOWN_OBSERVATION()] * depth), seq]);
            contexts += [np.array([tuple(seq_alt[ss:(ss+depth)]) for ss in range(len(seq))], dtype=list(zip([str(aa) for aa in range(0,depth+1)], ["O"] * depth)))];
        if(flatten):
            if(self.is_population):
                contexts = self._stack_arrays_by_subject(contexts)
            else:
                contexts = np.concatenate(contexts);

        return contexts;

    def setup_same_nback(self, depth : int, flatten : bool = True) -> np.ndarray:
        contexts = [];
        for seq in self.sequences:
            contexts += [[UNKNOWN_OBSERVATION()] * depth + list(seq[:-depth])];
        if(flatten):
            if(self.is_population):
                contexts = self._stack_arrays_by_subject([np.array(cc) for cc in contexts])
            else:
                contexts = np.concatenate([np.array(cc, dtype='object').flatten() for cc in contexts]);
        return contexts;


    def setup_local_time(self, flatten : bool = True) -> np.ndarray:
        times = [];
        for ss in self.session_lengths:
            times += [np.arange(1,ss+1)];
        if(flatten):
            if(self.is_population):
                times = np.concatenate(self._stack_arrays_by_subject(times))
            else:
                times = np.concatenate([np.array(cc, dtype='float').flatten() for cc in times]);
        return times;


    def get_within_session_timeconstant_labels(self) -> list[str]:
        if(not self.within_session_decay_enabled):
            return []
        elif(not self.distinct_session_within_session_timeconstants):
            return ["ALL"]
        else:
            return [str(ss) for ss in self.session_types]
        
    def setup_within_session_time_constant_ids(self, flatten : bool = True) -> np.ndarray:
        time_constant_id = [np.ones(ss,dtype=int) for ss in self.session_lengths]
        if(self.distinct_session_within_session_timeconstants):
            for ii, session_type in enumerate(self.session_types):
                for jj, label_c in enumerate(self.session_labels):
                    if(label_c == session_type):
                        time_constant_id[jj][:] = ii+1
        if(flatten):
            if(self.is_population):
                time_constant_id = np.concatenate(self._stack_arrays_by_subject(time_constant_id))
            else:
                time_constant_id = np.concatenate(time_constant_id);
        return time_constant_id


    def setup_session_ids(self, flatten : bool = True) -> np.ndarray:
        session_id = [np.ones(ss,dtype=int)+ii for ii,ss in enumerate(self.session_lengths)]
        if(flatten):
            if(self.is_population):
                session_id = np.concatenate(self._stack_arrays_by_subject(session_id))
            else:
                session_id = np.concatenate(session_id);
        return session_id

    def get_all_interaction_types(self) -> list[tuple]:
        interaction_types = []

        if(self.session_interactions_enabled):
            is_same_subject      = self.subject_labels[:,np.newaxis] == self.subject_labels[np.newaxis,:]
            is_before            = self.session_times[ :,np.newaxis] <  self.session_times[np.newaxis,:]
            possible_interaction = is_same_subject & is_before

            for aa in self.session_types:
                aa_i = np.where(self.session_labels == aa)[0]
                for bb in self.session_types:
                    bb_i = np.where(self.session_labels == bb)[0]

                    if(np.any(possible_interaction[aa_i,bb_i])):
                        interaction_types += [(aa,bb)]
        return interaction_types;

    def _to_interaction_name(self, types : tuple[str,str]) -> str:
        from_type = types[0]
        to_type   = types[1]
        return str(from_type).replace(' ', '_') + "_to_" + str(to_type).replace(' ', '_')


    def setup_session_interaction_times(self, concatenate : bool = True):
        interaction_timings = {}
        # column 1: projecting time, column 2: receiving time
        for from_type, to_type in self.get_all_interaction_types():
            interaction_name = "session_time_" + self._to_interaction_name((from_type, to_type))
            interaction_timings[interaction_name] = [np.zeros((ss, 2))-1  for ss in self.session_lengths];

            for session_num, session_type in enumerate(self.session_labels):
                if(session_type == from_type):
                    interaction_timings[interaction_name][session_num][:,0] = self.session_times[session_num]
                if(session_type == to_type):
                    interaction_timings[interaction_name][session_num][:,1] = self.session_times[session_num]

        if(concatenate):
            if(self.is_population):
                for kk in interaction_timings.keys():
                    interaction_timings[kk] = np.concatenate(self._stack_arrays_by_subject(interaction_timings[kk] )).astype(float)
            else:
                for kk in interaction_timings.keys():
                    interaction_timings[kk] = np.concatenate(interaction_timings[kk], axis=0).astype(float)
        return interaction_timings
             
    def _stack_arrays_by_subject(self, arrays : list[ArrayLike]):
        assert len(arrays) == self.num_sessions, "list of arrays must be the same as the number of sessions"
        
        C = [];
        for sub in self.subjects:
            C += [np.concatenate([np.array(arrays[ss]) for ss in np.where(self.subject_labels == sub)[0]])]
        return C;

    def _stack_arrays(self, arrays : list[ArrayLike],  T_0 : int = None):
        md = np.max([xx.ndim for xx in arrays])
        T = np.max([np.prod(xx.shape[1:]) for xx in arrays])
        if(not T_0 is None):
            assert T <= T_0, "found T is greater than expected T"
        N = np.sum([xx.shape[0] for xx in arrays])
        if(md > 1):
            C = np.zeros((N, T), dtype=arrays[0].dtype)
        else:
            C = np.zeros((N), dtype=arrays[0].dtype)

        ctr = 0;
        for ss in arrays:
            if(md > 1):
                C[ctr:ctr+ss.shape[0],:ss.shape[1]] = ss
            else:
                C[ctr:ctr+ss.shape[0]] = ss
            ctr += ss.shape[0];
        return C

    def get_population(self, sub):
        ii = self.subject_labels.tolist().index(sub)
        return self.populations.tolist().index(self.population_labels[ii])

    def setup_concatenated_start_times_per_subject(self) -> np.ndarray:
        return np.concatenate([[0],self.observations_per_subject]).astype(int).cumsum()
    
    def setup_population_dummies(self) -> np.ndarray:
        tts = self.setup_concatenated_start_times_per_subject();
        Z = np.zeros((self.total_observations, self.num_populations), dtype=int);
        for ii in range(self.num_subjects):
            Z[tts[ii]:tts[ii+1], self.get_population(self.subjects[ii])] = 1;
        return Z


    def to_dict(self) -> dict:
        
        data = {"N" : self.total_observations, "M" : self.M, "T" : np.max(self.observations_per_subject),
                "Y" : self.setup_compact_sequences(flatten=True),
                "K" : self.num_subjects,
                "local_time" : self.setup_local_time(flatten=True).astype(float),
                "local_timeconstant_id" : self.setup_within_session_time_constant_ids(flatten=True).astype(int),
                "session_id" : self.setup_session_ids(flatten=True).astype(int),
                "session_lengths" : self.session_lengths,
                "context_depth" : self.context_depth,
                "same_nback_depth" : self.same_nback_depth,
                "subject_start_idx" : self.setup_concatenated_start_times_per_subject()+1,}
                # "subject_labels" : self.subject_labels,
                # "session_labels" : self.session_labels,
        
        for depth in range(1, self.context_depth+1):
            contexts = self.setup_contexts(depth);
            if(not self.is_population):
                match = contexts[:,np.newaxis] == contexts[np.newaxis,:]
                data[f"is_same_context_{depth}"] = np.tril(match,-1).astype(int)
            else:
                data[f"is_same_context_{depth}"] = self._stack_arrays([np.tril(contexts_c[:,np.newaxis] == contexts_c[np.newaxis,:],-1).astype(int) for contexts_c in contexts])

        base = self.flatten_sequences();
        for depth in range(1, self.same_nback_depth+1):
            nback = self.setup_same_nback(depth);
            if(not self.is_population):
                match = nback[:,np.newaxis] == base[np.newaxis,:]
                data[f"is_same_{depth}_back"] = np.tril(match).astype(int)
            else:
                data[f"is_same_{depth}_back"] = self._stack_arrays([np.tril(nback_c[:,np.newaxis] == base_c[np.newaxis,:]).astype(int) for nback_c, base_c in zip(nback, base)])


        if(self.is_population_partial):
            Z = self.setup_population_dummies();
            if(not self.population_shared_alpha):
                data["P_alpha"] = self.num_populations
                data["alpha_loadings"] = Z
            else:
                data["P_alpha"] = 1
                data["alpha_loadings"] = np.ones((self.total_observations,1),dtype=int)

            if(not self.population_shared_within_session_timescale):
                data["P_within_session_timeconstants"] = self.num_populations
                for ii in self.get_within_session_timeconstant_labels():
                    data[f"timeconstant_within_session_{ii}_loadings"] = Z
            else:
                data["P_within_session_timeconstants"] = 1
                for ii in self.get_within_session_timeconstant_labels():
                    data[f"timeconstant_within_session_{ii}_loadings"] = np.ones((self.total_observations,1),dtype=int)


            if(not self.population_shared_between_session_timescale):
                data["P_between_session_timeconstants"] = self.num_populations
                for ii in self.get_all_interaction_types():
                    data[f"timeconstant_between_sessions_{ii}_loadings"] = Z
            else:
                data["P_between_session_timeconstants"] = 1
                for ii in self.get_all_interaction_types():
                    data[f"timeconstant_between_sessions_{ii}_loadings"] = np.ones((self.total_observations,1),dtype=int)

            for ii in range(self.context_depth):
                if(not self.population_shared_context[ii]):
                    data[f"P_context_similarity_depth_{ii+1}"] = self.num_populations
                    data[f"context_similarity_depth_{ii+1}_loadings"] = Z
                else:
                    data[f"P_context_similarity_depth_{ii+1}"] = 1
                    data[f"context_similarity_depth_{ii+1}_loadings"] = np.ones((self.total_observations,1),dtype=int)

            for ii in range(self.same_nback_depth):
                if(not self.population_shared_same_nback[ii]):
                    data[f"P_repeat_bias_{ii+1}_back"] = self.num_populations
                    data[f"repeat_bias_{ii+1}_back_loadings"] = Z
                else:
                    data[f"P_repeat_bias_{ii+1}_back"] = 1
                    data[f"repeat_bias_{ii+1}_back_loadings"] = np.ones((self.total_observations,1),dtype=int)

        if(not (self.fixed_alpha is None)):
            data["alpha"] = self.fixed_alpha;

        data.update(self.setup_session_interaction_times())

        data.update(self.priors.to_dict())
        return data
    

    @property
    def data(self):
        return self.to_dict();

    @property
    def model(self):
        pop_model = self.is_population
        multisession_model = np.max(self.sessions_per_subject) > 1
        context_depth = self.context_depth
        repeat_depth = self.same_nback_depth

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


        session_interaction_types    = self.get_all_interaction_types()
        within_session_timeconstants = self.get_within_session_timeconstant_labels()


        if(self.is_population_partial):
            return stanModels.generate_stan_code_population_shared_parameters(session_interaction_types=session_interaction_types,
                                                            within_session_timeconstants=within_session_timeconstants,
                                                            context_depth=self.context_depth,
                                                            same_nback_depth=self.same_nback_depth);

        else:
            return stanModels.generate_stan_code_individual(session_interaction_types=session_interaction_types,
                                                            within_session_timeconstants=within_session_timeconstants,
                                                            context_depth=self.context_depth,
                                                            same_nback_depth=self.same_nback_depth,
                                                            fit_alpha=(self.fixed_alpha is None));


    def build(self, random_seed : int) -> stan.model.Model:
        self.posterior = stan.build(self.model, data=self.data, random_seed=int(random_seed))

    def fit_model(self, num_chains : int = 4, num_samples : int = 1000, random_seed : int = None) -> stan.fit.Fit:
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
        func_dict = {
            "median": np.median,
            "2.5%": lambda x: np.percentile(x, 2.5),
            "5.0%": lambda x: np.percentile(x, 5.0),
            "10.0%": lambda x: np.percentile(x, 10.0),
            "25.0%": lambda x: np.percentile(x, 25.0),

            "97.5%": lambda x: np.percentile(x, 97.5),
            "95.0%": lambda x: np.percentile(x, 95.0),
            "90.0%": lambda x: np.percentile(x, 90.0),
            "75.0%": lambda x: np.percentile(x, 75.0),
        }
        sum_df =  az.summary(self.fit, stat_funcs=func_dict).sort_index()
        s = sum_df.loc["log_like":"log_likf"].index

        return sum_df.drop(index=s)
    
    def waic(self, **kwargs) -> az.ELPDData:
        if(self.fit is None):
            raise ValueError("Model not fit yet!")

        c = az.convert_to_inference_data(self.fit)
        c.sample_stats["log_likelihood"] = c.posterior["log_likelihood"]
        return az.waic(c, **kwargs)
    
    def loo(self, **kwargs) -> az.ELPDData:
        if(self.fit is None):
            raise ValueError("Model not fit yet!")

        c = az.convert_to_inference_data(self.fit)
        c.sample_stats["log_likelihood"] = c.posterior["log_likelihood"]
        return az.loo(c, **kwargs)

    
    def log_prob(self, unconstrained_parameters: ArrayLike, adjust_transform: bool = True, return_inf_at_error : bool = True, print_error : bool = False) -> float:
        if(np.any(np.isnan(unconstrained_parameters))):
            raise ValueError("NaNs found in parameters: " + str(unconstrained_parameters))
        if(np.any(np.isinf(unconstrained_parameters))):
            raise ValueError("infs found in parameters: " + str(unconstrained_parameters))
        
        try:
            f =  self.posterior.log_prob(unconstrained_parameters=list(unconstrained_parameters),
                                        adjust_transform=adjust_transform);
            return f;
        except:
            if(print_error):
                print("Error in log_prob with params: " + str(unconstrained_parameters))
            if(return_inf_at_error):
                return -np.inf;
            else:
                raise

    
    def grad_log_prob(self, unconstrained_parameters: ArrayLike, adjust_transform: bool = True, return_inf_at_error : bool = True, print_error : bool = False) -> float:
        """Calculate the gradient of the log posterior evaluated at
            the unconstrained parameters.

        Arguments:
            unconstrained_parameters: A sequence of unconstrained parameters.

        Returns:
            The gradient of the log posterior evaluated at the
            unconstrained parameters.

        Notes:
            The unconstrained parameters are passed to the log_prob_grad
            function in stan::model.
        """
        if(np.any(np.isnan(unconstrained_parameters))):
            raise ValueError("NaNs found in parameters: " + str(unconstrained_parameters))
        if(np.any(np.isinf(unconstrained_parameters))):
            raise ValueError("infs found in parameters: " + str(unconstrained_parameters))

        try:
            payload = {
                "data": self.posterior.data,
                "unconstrained_parameters": list(unconstrained_parameters),
                "adjust_transform": adjust_transform,
            }

            async def go():
                async with stan.common.HttpstanClient() as client:
                    resp = await client.post(f"/{self.posterior.model_name}/log_prob_grad", json=payload)
                    if resp.status != 200:
                        raise RuntimeError(resp.json())
                    return resp.json()["log_prob_grad"]

            xx = asyncio.run(go())
            return np.array(xx)
        except:
            if(print_error):
                print("Error in grad_log_prob with params: " + str(unconstrained_parameters))
            if(return_inf_at_error):
                xx = np.sign(unconstrained_parameters)*-1.0;
                return xx
            else:
                raise
        
    
    def get_map(self, x_0 : ArrayLike = None, adjust_transform: bool = False):
        if(self.posterior is None):
            raise ValueError("Model not built yet!")
        
        if(x_0 is None):
            x_0 = np.zeros((len(self.posterior.param_names)), dtype=float)
        
        nll  = lambda ps : -self.log_prob(ps, adjust_transform=adjust_transform)
        dnll = lambda ps : -self.grad_log_prob(ps, adjust_transform=adjust_transform)

        map_fit = minimize(nll, x_0,  jac=dnll)
        # print(map_fit)
        x_fit   = self.posterior.constrain_pars(list(map_fit.x))

        return dict(map(lambda i,j : (i,j) , self.posterior.param_names, x_fit))