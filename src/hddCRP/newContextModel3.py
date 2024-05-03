import numpy as np
import numpy.typing as npt
import stan, stan.model, stan.fit
import pandas as pd
from hddCRP import dataLoader as dl
from hddCRP.markovModeling import get_sequence_likelihood_with_condition
import pickle

class populationTurnLocationContextModel:
    def __init__(self, Y : npt.NDArray[np.int_], R : npt.NDArray[np.int_], subject_groups : list, base_measure : npt.NDArray = None):
        assert Y.ndim == 2, "Y must be 2-D"
        # assert Y.shape[1] > 1, "currently, needs to be population model"
        self.Y_raw_ = Y
        self.symbols_,self.Y_ = np.unique(Y, return_inverse=True);
        self.Y_ = self.Y_.reshape(Y.shape)+1

        self.R_ = np.array(R,dtype=int).reshape(self.Y_.shape)+1

        self.prior_context_b_depth_1_alpha_ = 1.0
        self.prior_context_b_depth_1_beta_  = 1.0
        self.prior_context_a_depth_1_alpha_ = 1.0
        self.prior_context_a_depth_1_beta_  = 1.0
        self.prior_subject_similarity_alpha_ = 1.0
        self.prior_subject_similarity_beta_ = 1.0

        self.subject_groups_0_ = subject_groups;
        _, self.subject_groups_ = np.unique(np.array(subject_groups).ravel(), return_inverse=True);
        self.subject_groups_ += 1;
        assert (self.subject_groups_.size) == self.K, f"subject_groups wrong size. Received {self.subject_groups_.size}, expected {self.K}"

        self.prior_alpha_shape_ = 2.0
        if(self.K == 1):
            self.fit_base_measure_ = False
            self.prior_alpha_scale_ = 2.0 
        else:
            self.fit_base_measure_ = True
            self.prior_alpha_scale_ = 2.0 / self.M

        self.prior_timeconstant_within_session_shape_ = 2.0
        self.prior_timeconstant_within_session_scale_ = 20.0

        if(base_measure is None):
            base_measure = np.ones((self.M))

        self.base_measure_ = np.array(base_measure).ravel()
        self.base_measure_ = self.base_measure_/np.sum(self.base_measure_)
        assert self.base_measure_.size == self.M, f"base measure wrong size. Received {self.base_measure_.size}, expected {self.M}"
        assert np.all(self.base_measure_>0), "base measure must be all positive"

        # self.setup_cpu_computation_()
        self.posterior = None

    def build(self, random_seed : int) -> stan.model.Model:
        if(self.fit_base_measure_ ):
            stan_model = self.model;
        else:
            stan_model = self.model_fixed_base;
        self.posterior = stan.build(stan_model, data=self.data_dict(), random_seed=int(random_seed))

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


    @property
    def M(self) -> int:
        return self.symbols_.size
    @property
    def K(self) -> int:
        return self.Y_.shape[1]
    @property
    def N(self) -> int:
        return self.Y_.shape[0]
    @property
    def Y(self) -> npt.NDArray[np.int_]:
        return self.Y_.copy()
    @property
    def R(self) -> npt.NDArray[np.int_]:
        return self.R_.copy()


   

    def data_dict(self) -> dict:

        data = {"N" : self.N, "M" : self.M, "K" : self.K,
                "Y" : self.Y_,
                "R" : self.R_,
                "groups" : self.subject_groups_,
                "base_measure" : self.base_measure_,
                "prior_alpha_shape" : self.prior_alpha_shape_,
                "prior_alpha_scale" : self.prior_alpha_scale_,
                "prior_timeconstant_within_session_shape" : self.prior_timeconstant_within_session_shape_,
                "prior_timeconstant_within_session_scale" : self.prior_timeconstant_within_session_scale_,
                "prior_context_b_depth_1_alpha" : self.prior_context_b_depth_1_alpha_,
                "prior_context_b_depth_1_beta" : self.prior_context_b_depth_1_beta_,
                "prior_context_a_depth_1_alpha" : self.prior_context_a_depth_1_alpha_,
                "prior_context_a_depth_1_beta" : self.prior_context_a_depth_1_beta_,
                "prior_subject_similarity_alpha" : self.prior_subject_similarity_alpha_,
                "prior_subject_similarity_beta" : self.prior_subject_similarity_beta_,
                }
        return data
    @property
    def model(self) -> str:
        return """
data {
    int N; // Number of data points
    int M; // number of possible observations
    int K; // number of subjects that are stacked on top of each other
    array[N, K] int Y;
    array[N, K] int R;
    array[K] int groups;

    vector[M] base_measure;

    real prior_alpha_shape;
    real prior_alpha_scale;

    real prior_timeconstant_within_session_shape;
    real prior_timeconstant_within_session_scale;

    real prior_context_a_depth_1_alpha;
    real prior_context_a_depth_1_beta;
    real prior_context_b_depth_1_alpha;
    real prior_context_b_depth_1_beta;

    real prior_subject_similarity_alpha;
    real prior_subject_similarity_beta;
}

transformed data {
    matrix[N*K,N*K] valid    = rep_matrix(0, N*K, N*K);
    matrix[N*K,N*K] is_same  = rep_matrix(0, N*K, N*K);
    matrix[N*K,N*K] delta_ts = rep_matrix(0, N*K, N*K);

    matrix[N*K,N*K] different_subjects_a = rep_matrix(0, N*K, N*K);
    matrix[N*K,N*K] different_subjects_b = rep_matrix(0, N*K, N*K);

    matrix[N*K,N*K] X_context_a_depth_1 = rep_matrix(0, N*K, N*K);
    matrix[N*K,N*K] X_context_b_depth_1 = rep_matrix(0, N*K, N*K);

    matrix[N*K,M] base = rep_matrix(0, N*K, M);

    for(jj in 1:K) {
        int oo_a = (jj-1)*N;
        for(aa in 1:N) {
            base[oo_a + aa, Y[aa,jj]] = base_measure[Y[aa,jj]];
        }

        for(kk in 1:K) {
            int oo_b = (kk-1)*N;
            for(aa in 1:N) {
                for(bb in 1:(aa-1)) {
                    delta_ts[oo_a+aa, oo_b+bb] = aa-bb;
                    valid[oo_a+aa, oo_b+bb] = 1;

                    if(jj != kk) {
                        different_subjects_a[oo_a+aa, oo_b+bb] = 1;
                    }
                    if(groups[jj] != groups[kk]) {
                        different_subjects_b[oo_a+aa, oo_b+bb] = 1;
                    }

                    if(Y[aa,jj] == Y[bb,kk]) {
                        is_same[oo_a+aa, oo_b+bb] = 1;
                    }

                    int r_match = 0;
                    int Y_match_1 = 0;

                    if(R[aa,jj] == R[bb,kk]) {
                        r_match = 1;
                    }

                    if(aa > 1 && bb > 1) {
                        if(Y[aa-1,jj] == Y[bb-1,kk]) {
                            Y_match_1 = 1;
                        }
                    }

                    if(r_match == 0) {
                        X_context_a_depth_1[oo_a+aa, oo_b+bb] = 1;
                    }
                    if(Y_match_1 == 0 ){
                        X_context_b_depth_1[oo_a+aa, oo_b+bb] = 1;
                    }

                }
            }
        }
    }

    // prior parameter transformations for computations
    real prior_alpha_scale_inv = inv(prior_alpha_scale);

    real prior_timeconstant_within_session_scale_inv   = inv(prior_timeconstant_within_session_scale);

    // variables to turn main computation in matrix operations
    vector[N*K] vs = rep_vector(1, N*K);
}

parameters {
    vector<lower=0>[M] alpha;
    real<lower=0> timeconstant;
    real<lower=0,upper=1> subject_similarity_a;
    real<lower=0,upper=1> subject_similarity_b;
    real<lower=0,upper=1> context_a_depth_1;
    real<lower=0,upper=1> context_b_depth_1;
}
model {

    alpha         ~ gamma(prior_alpha_shape,                       prior_alpha_scale_inv);
    timeconstant  ~ gamma(prior_timeconstant_within_session_shape, prior_timeconstant_within_session_scale_inv);
    context_a_depth_1     ~ beta(prior_context_a_depth_1_alpha,  prior_context_a_depth_1_beta);
    context_b_depth_1     ~ beta(prior_context_b_depth_1_alpha,  prior_context_b_depth_1_beta);

    subject_similarity_a  ~ beta(prior_subject_similarity_alpha, prior_subject_similarity_beta);
    subject_similarity_b  ~ beta(prior_subject_similarity_alpha, prior_subject_similarity_beta);

    vector[N*K] ps;
    matrix[N*K,N*K] weights_same_obs;
    matrix[N*K,N*K] weights_all_obs;
    weights_all_obs = -delta_ts./timeconstant
                      + log(subject_similarity_a)       * different_subjects_a
                      + log(subject_similarity_b)       * different_subjects_b
                      + log1m(context_a_depth_1) * X_context_a_depth_1
                      + log1m(context_b_depth_1) * X_context_b_depth_1;
    weights_all_obs   = valid .* exp(weights_all_obs);

    weights_same_obs  = is_same .* weights_all_obs;

    // probability of drawing the observed observations given their pasts
    ps =  ((weights_same_obs * vs) + base*alpha) ./  ((weights_all_obs * vs) + dot_product(alpha,base_measure));

    1 ~ bernoulli(ps); // note: not generative - this is a simplified distribution to make the log likelihood computations work quickly in Stan
}
"""
    @property
    def model_fixed_base(self) -> str:
        return """
data {
    int N; // Number of data points
    int M; // number of possible observations
    int K; // number of subjects that are stacked on top of each other
    array[N, K] int Y;
    array[N, K] int R;
    array[K] int groups;

    array[M] real base_measure;

    real prior_alpha_shape;
    real prior_alpha_scale;

    real prior_timeconstant_within_session_shape;
    real prior_timeconstant_within_session_scale;

    real prior_context_a_depth_1_alpha;
    real prior_context_a_depth_1_beta;
    real prior_context_b_depth_1_alpha;
    real prior_context_b_depth_1_beta;

    real prior_subject_similarity_alpha;
    real prior_subject_similarity_beta;
}

transformed data {
    matrix[N*K,N*K] valid    = rep_matrix(0, N*K, N*K);
    matrix[N*K,N*K] is_same  = rep_matrix(0, N*K, N*K);
    matrix[N*K,N*K] delta_ts = rep_matrix(0, N*K, N*K);

    vector[N*K] base = rep_vector(0, N*K);
    real G = sum(base_measure);

    matrix[N*K,N*K] different_subjects_a = rep_matrix(0, N*K, N*K);
    matrix[N*K,N*K] different_subjects_b = rep_matrix(0, N*K, N*K);

    matrix[N*K,N*K] X_context_a_depth_1 = rep_matrix(0, N*K, N*K);
    matrix[N*K,N*K] X_context_b_depth_1 = rep_matrix(0, N*K, N*K);


    for(jj in 1:K) {
        int oo_a = (jj-1)*N;
        for(aa in 1:N) {
            base[oo_a + aa] = base_measure[Y[aa,jj]];
        }

        for(kk in 1:K) {
            int oo_b = (kk-1)*N;
            for(aa in 1:N) {
                for(bb in 1:(aa-1)) {
                    delta_ts[oo_a+aa, oo_b+bb] = aa-bb;
                    valid[oo_a+aa, oo_b+bb] = 1;

                    if(jj != kk) {
                        different_subjects_a[oo_a+aa, oo_b+bb] = 1;
                    }
                    if(groups[jj] != groups[kk]) {
                        different_subjects_b[oo_a+aa, oo_b+bb] = 1;
                    }
                    if(Y[aa,jj] == Y[bb,kk]) {
                        is_same[oo_a+aa, oo_b+bb] = 1;
                    }

                    int r_match = 0;
                    int Y_match_1 = 0;
                    if(R[aa,jj] == R[bb,kk]) {
                        r_match = 1;
                    }

                    if(aa > 1 && bb > 1) {
                        if(Y[aa-1,jj] == Y[bb-1,kk]) {
                            Y_match_1 = 1;
                        }
                    }

                    if(r_match == 0) {
                        X_context_a_depth_1[oo_a+aa, oo_b+bb] = 1;
                    }
                    if(Y_match_1 == 0 ){
                        X_context_b_depth_1[oo_a+aa, oo_b+bb] = 1;
                    }

                }
            }
        }
    }

    // prior parameter transformations for computations
    real prior_alpha_scale_inv = inv(prior_alpha_scale);

    real prior_timeconstant_within_session_scale_inv   = inv(prior_timeconstant_within_session_scale);

    // variables to turn main computation in matrix operations
    vector[N*K] vs = rep_vector(1, N*K);
}

parameters {
    real<lower=0> alpha;
    real<lower=0> timeconstant;
    real<lower=0,upper=1> context_a_depth_1;
    real<lower=0,upper=1> context_b_depth_1;
    real<lower=0,upper=1> subject_similarity_a;
    real<lower=0,upper=1> subject_similarity_b;
}
model {

    alpha         ~ gamma(prior_alpha_shape,                       prior_alpha_scale_inv);
    timeconstant  ~ gamma(prior_timeconstant_within_session_shape, prior_timeconstant_within_session_scale_inv);
    context_a_depth_1     ~ beta(prior_context_a_depth_1_alpha,  prior_context_a_depth_1_beta);
    context_b_depth_1     ~ beta(prior_context_b_depth_1_alpha,  prior_context_b_depth_1_beta);

    subject_similarity_a  ~ beta(prior_subject_similarity_alpha, prior_subject_similarity_beta);
    subject_similarity_b  ~ beta(prior_subject_similarity_alpha, prior_subject_similarity_beta);

    vector[N*K] ps;
    matrix[N*K,N*K] weights_same_obs;
    matrix[N*K,N*K] weights_all_obs;
    weights_all_obs = -delta_ts./timeconstant
                      + log1m(context_a_depth_1) * X_context_a_depth_1
                      + log1m(context_b_depth_1) * X_context_b_depth_1
                      + log(subject_similarity_a)       * different_subjects_a
                      + log(subject_similarity_b)       * different_subjects_b;
    weights_all_obs   = valid .* exp(weights_all_obs);

    weights_same_obs  = is_same .* weights_all_obs;

    // probability of drawing the observed observations given their pasts
    ps =  ((weights_same_obs * vs) + alpha*base) ./  ((weights_all_obs * vs) + alpha*G);

    1 ~ bernoulli(ps); // note: not generative - this is a simplified distribution to make the log likelihood computations work quickly in Stan
}
"""

def load_turns_and_locations(sub):
    data_filename = 'michelle/Data_location_visits_phase2_v03.pkl';
    with open(data_filename, 'rb') as data_file:
        data_file = pickle.load(data_file)
    for grp in data_file["data_by_session"].keys():
        sname = sub + "_" + grp
        keys = data_file["data_by_session"][grp].keys()

        # print(f"{sname} : {keys}")
        if(sname in keys):
            seqs = [np.array(xx,dtype=int) for xx in data_file["data_by_session"][grp][sname]]

            R = [xx[:-1] for xx in seqs]
            Y = [np.zeros(xx.size-1,dtype=int) for xx in seqs]

            for ii in range(len(seqs)):
                starts = seqs[ii][:-1]
                ends   = seqs[ii][1:]

                ls = ((starts + 1) % 4) == ends
                rs = ((starts - 1) % 4) == ends
                ss = ((starts + 2) % 4) == ends

                Y[ii][ls] = 0
                Y[ii][ss] = 1
                Y[ii][rs] = 2

                return Y,R
    return None        


def create_pop_model(grp : str, n_trials : int = 50, fold_turns : bool = False, last : bool = False) -> populationTurnLocationContextModel:
    subs = dl.get_subjects(grp);
    Y = np.ndarray((n_trials, len(subs)), dtype=int)
    R = np.ndarray((n_trials, len(subs)), dtype=int)
    grps = [];

    for ii,sub in enumerate(subs):
        grps.append(dl.get_group(sub))
        seqs,rs = load_turns_and_locations(sub)
        seqs = np.concatenate(seqs);
        rs = np.concatenate(rs);
        if(last):
            Y[:,ii] = seqs[-n_trials:];
            R[:,ii] = rs[-n_trials:];
        else:
            Y[:,ii] = seqs[:n_trials];
            R[:,ii] = rs[:n_trials];

    if(fold_turns):
        Y[Y == 2] = 0
        base_measure = np.array([2, 1]);
    else:
        base_measure = np.ones((3))

    return populationTurnLocationContextModel(Y, R, subject_groups=grps, base_measure=base_measure)

def create_indiv_model(sub : str, n_trials : int = 50, fold_turns : bool = False, last : bool = False, use_turns : bool = True, use_recoded_turns : bool = False) -> populationTurnLocationContextModel:

    grps = [];

    grps.append(dl.get_group(sub))
    seqs,rs = dl.load_raw_with_reward_phase2(sub, use_turns=use_turns, use_recoded_turns=use_recoded_turns)
    seqs = np.concatenate(seqs);
    rs = np.concatenate(rs);

    Y = np.ndarray((n_trials, 1), dtype=int)
    R = np.ndarray((n_trials, 1), dtype=int)
    if(last):
        Y[:,0] = seqs[-n_trials:];
        R[:,0] = rs[-n_trials:];
    else:
        Y[:,0] = seqs[:n_trials];
        R[:,0] = rs[:n_trials];

    if(fold_turns):
        Y[Y == 2] = 0
        base_measure = np.array([2, 1]);
    elif(use_turns):
        base_measure = np.ones((3))
    else:
        base_measure = np.ones((4))

    return populationTurnLocationContextModel(Y, R, subject_groups=grps, base_measure = base_measure)