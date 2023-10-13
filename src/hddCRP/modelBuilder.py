
import numpy as np;
from numpy.typing import ArrayLike
import stan

class UNKNOWN_OBSERVATION:
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return True
        else:
            return False

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

        self.nback_depth = nback_depth;
        self.context_depth = context_depth

    @property
    def num_sessions(self) -> int:
        return len(self.sequences);

    @property
    def session_lengths(self) -> np.ndarray[int]:
        return np.array([len(ss) for ss in self.sequences], dtype=int);

    @property
    def observations_per_subject(self) -> int:
        return np.array([np.sum(self.session_lengths[self.subject_labels == sub]) for sub in self.subjects])
    

    @property
    def session_types(self) -> list[int]:
        return np.unique(self.session_labels);

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
    @property
    def nback_depth(self) -> int:
        return self._nback_depth;
    @nback_depth.setter
    def nback_depth(self, nd : int) -> None:
        nd = int(nd)
        assert nd > 0, "nback_depth must be non-negative integer"
        self._nback_depth = nd
    
    def setup_compact_sequences(self, flatten=True) -> list[np.ndarray[int]]:
        v = np.arange(1,self.M+1).astype(int);
        seqs = [(np.array(ss)[:,np.newaxis] == self.possible_observations[np.newaxis,:]) @ v for ss in self.sequences]
        if(flatten):
            seqs = np.concatenate(seqs);
        return seqs;

    def flatten_sequences(self, flatten=True) -> list[np.ndarray[int]]:
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

        

