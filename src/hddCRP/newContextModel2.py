import numpy as np
import numpy.typing as npt
import stan, stan.model, stan.fit
import arviz as az
import pandas as pd
from hddCRP import dataLoader as dl
from hddCRP.markovModeling import get_sequence_likelihood_with_condition

class populationOneBackRewardContextModel:
    def __init__(self, Y : npt.NDArray[np.int_], R : npt.NDArray[np.int_], base_measure : npt.NDArray = None):
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

        self.setup_cpu_computation_()

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

    def setup_cpu_computation_(self):
        self.Y_1_back_ = np.roll(self.Y.astype(float),1,axis=0)
        self.Y_1_back_[0,:] = np.nan
        self.R_1_back_ = np.roll(self.R.astype(float),1,axis=0)
        self.R_1_back_[0,:] = np.nan

        self.match_ = np.zeros((self.N, self.N, self.K, self.K), dtype=bool)
        self.different_subject_ = np.zeros((self.N, self.N, self.K, self.K), dtype=float)
        self.is_context_a_ = np.zeros((self.N, self.N, self.K, self.K), dtype=float)
        self.is_context_b_ = np.zeros((self.N, self.N, self.K, self.K), dtype=float)
        for jj in range(self.K):
            for kk in range(self.K):
                self.is_context_a_[:,:,jj,kk] = (self.R_1_back_[:,[jj]] != self.R_1_back_[:,[kk]].T) # (self.Y_1_back_[:,[jj]] == self.Y_1_back_[:,[kk]].T) & 
                self.is_context_b_[:,:,jj,kk] = (self.Y_1_back_[:,[jj]] != self.Y_1_back_[:,[kk]].T)
                self.different_subject_[:,:,jj,kk] = jj != kk
                self.match_[:,:,jj,kk] = (self.Y_[:,[jj]] == self.Y_[:,[kk]].T)

    def compute_log_likelihood(self, timescale : float, context_a : float, context_b : float,
                               alpha : npt.NDArray[np.float_] | float, subject_similarity : float) -> npt.NDArray[np.float_]:
        lls = np.zeros((self.N, self.K))

        tau = (np.arange(-self.N,0)/timescale).reshape((self.N,1))

        alphas = alpha[self.Y-1]
        G = np.sum(alpha)

        if(context_a is None):
            context_a = np.zeros_like(self.is_context_a_)
        elif(context_a >= 1):
            context_a = np.zeros_like(self.is_context_a_)
            context_a[self.is_context_a_ > 0] = -np.inf
        elif(context_a <= 0):
            context_a = np.zeros_like(self.is_context_a_)
        else:
            context_a = np.log1p(-context_a)*self.is_context_a_;

        if(context_b is None):
            context_b = np.zeros_like(self.is_context_b_)
        elif(context_b >= 1):
            context_b = np.zeros_like(self.is_context_b_)
            context_b[self.is_context_b_ > 0] = -np.inf
        elif(context_b <= 0):
            context_b = np.zeros_like(self.is_context_b_)
        else:
            context_b = np.log1p(-context_b)*self.is_context_b_;

        if(subject_similarity <= 0 or subject_similarity is None):
            subject_similarity = np.zeros_like(self.different_subject_)
            subject_similarity[self.different_subject_ > 0] = -np.inf
        elif(subject_similarity >= 1):
            subject_similarity = np.zeros_like(self.different_subject_)
        else:
            subject_similarity = np.log(subject_similarity) * self.different_subject_


        Z = context_a + context_b + subject_similarity;

        lls[0,:] = np.log(alphas[0,:]) - np.log(G)
        for tt in range(1,self.N):
            for ss in range(self.K):
                X = np.zeros((tt,self.K))
                # X += context_a * self.is_context_a_[tt,:tt,ss,:]
                # X += context_b * self.is_context_b_[tt,:tt,ss,:]
                # X += subject_similarity * self.different_subject_[tt,:tt,ss,:]
                X += Z[tt,:tt,ss,:]
                X += tau[-tt:,:]
                X = np.exp(X)
                lls[tt,ss] = np.log(np.sum(X*self.match_[tt,:tt,ss,:]) + alphas[tt,ss]) - np.log(np.sum(X) + G)
        return lls

    def compute_cross_val_test(self, context_a : float, context_b : float,
                               alpha : float, alpha_init : float):



        P = np.zeros((2, self.M, self.M,  self.K))
        Q = np.zeros(( self.M,  self.K))

        for kk in range(self.K):
            Y = self.Y_[:,kk]
            Y_1 = self.Y_1_back_[:,kk]
            R_1 = self.R_1_back_[:,kk]
            for ii in range(self.M):
                Q[ii,kk] = np.sum(self.Y_[:,kk] == ii+1);

                for jj in range(self.M):
                    for rr in range(2):
                        P[rr,jj,ii,kk] = np.sum((Y==ii+1) & (Y_1 == jj+1) & (R_1 == rr+1))


        ll_seq = np.zeros((self.K))
        for kk in range(self.K):
            P_c = np.sum(P[:,:,:, [xx for xx in range(self.K) if xx != kk]],axis=-1)

            M = np.zeros_like(P_c)
            for rr in range(2):
                for jj in range(self.M):
                    oo = [xx for xx in range(self.M) if xx != jj];

                    M[rr,jj,:] = P_c[rr,jj,:]
                    M[rr,jj,:] += (1-context_b)* P_c[rr,oo,:].sum(axis=0)
                    M[rr,jj,:] += (1-context_b)* P_c[(rr + 1) % 2,oo,:].sum(axis=0) 
                    M[rr,jj,:] += (1-context_a)* P_c[(rr + 1) % 2,jj,:] 
                    M[rr,jj,:] +=  alpha

            T = M / np.sum(M,axis=-1,keepdims=True)

            Y = self.Y_[:,kk]-1
            R = self.R_[:,kk]-1

            ll_seq[kk] = get_sequence_likelihood_with_condition(Y, R, T)

            Q_c = np,
            ll_start = np.sum(Q[:,[xx for xx in range(self.K) if xx != kk]],axis=1)
            ll_start = np.log(ll_start[Y[0]] + alpha_init/self.M) - np.log(np.sum(ll_start) + alpha_init);
        return ll_seq

    def data_dict(self) -> dict:

        data = {"N" : self.N, "M" : self.M, "K" : self.K,
                "Y" : self.Y_,
                "R" : self.R_,
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

    matrix[N*K,N*K] different_subjects = rep_matrix(0, N*K, N*K);

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
                        different_subjects[oo_a+aa, oo_b+bb] = 1;
                    }
                    if(Y[aa,jj] == Y[bb,kk]) {
                        is_same[oo_a+aa, oo_b+bb] = 1;
                    }

                    int r_match = 0;
                    int Y_match_2 = 0;
                    int Y_match_1 = 0;

                    if(aa > 1 && bb > 1) {
                        if(Y[aa-1,jj] == Y[bb-1,kk]) {
                            Y_match_1 = 1;
                        }
                        if(R[aa-1,jj] == R[bb-1,kk]) {
                            r_match = 1;
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
    real<lower=0,upper=1> subject_similarity;
    real<lower=0,upper=1> context_a_depth_1;
    real<lower=0,upper=1> context_b_depth_1;
}
model {

    alpha         ~ gamma(prior_alpha_shape,                       prior_alpha_scale_inv);
    timeconstant  ~ gamma(prior_timeconstant_within_session_shape, prior_timeconstant_within_session_scale_inv);
    context_a_depth_1     ~ beta(prior_context_a_depth_1_alpha,  prior_context_a_depth_1_beta);
    context_b_depth_1     ~ beta(prior_context_b_depth_1_alpha,  prior_context_b_depth_1_beta);

    subject_similarity    ~ beta(prior_subject_similarity_alpha, prior_subject_similarity_beta);

    vector[N*K] ps;
    matrix[N*K,N*K] weights_same_obs;
    matrix[N*K,N*K] weights_all_obs;
    weights_all_obs = -delta_ts./timeconstant
                      + log(subject_similarity)       * different_subjects
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

    matrix[N*K,N*K] different_subjects = rep_matrix(0, N*K, N*K);

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
                        different_subjects[oo_a+aa, oo_b+bb] = 1;
                    }
                    if(Y[aa,jj] == Y[bb,kk]) {
                        is_same[oo_a+aa, oo_b+bb] = 1;
                    }

                    int r_match = 0;
                    int Y_match_2 = 0;
                    int Y_match_1 = 0;

                    if(aa > 1 && bb > 1) {
                        if(Y[aa-1,jj] == Y[bb-1,kk]) {
                            Y_match_1 = 1;
                        }
                        if(R[aa-1,jj] == R[bb-1,kk]) {
                            r_match = 1;
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
    real<lower=0,upper=1> subject_similarity;
    real<lower=0,upper=1> context_a_depth_1;
    real<lower=0,upper=1> context_b_depth_1;
}
model {

    alpha         ~ gamma(prior_alpha_shape,                       prior_alpha_scale_inv);
    timeconstant  ~ gamma(prior_timeconstant_within_session_shape, prior_timeconstant_within_session_scale_inv);
    context_a_depth_1     ~ beta(prior_context_a_depth_1_alpha,  prior_context_a_depth_1_beta);
    context_b_depth_1     ~ beta(prior_context_b_depth_1_alpha,  prior_context_b_depth_1_beta);

    subject_similarity    ~ beta(prior_subject_similarity_alpha, prior_subject_similarity_beta);

    vector[N*K] ps;
    matrix[N*K,N*K] weights_same_obs;
    matrix[N*K,N*K] weights_all_obs;
    weights_all_obs = -delta_ts./timeconstant
                      + log(subject_similarity)  * different_subjects
                      + log1m(context_a_depth_1) * X_context_a_depth_1
                      + log1m(context_b_depth_1) * X_context_b_depth_1;
    weights_all_obs   = valid .* exp(weights_all_obs);

    weights_same_obs  = is_same .* weights_all_obs;

    // probability of drawing the observed observations given their pasts
    ps =  ((weights_same_obs * vs) + alpha*base) ./  ((weights_all_obs * vs) + alpha*G);

    1 ~ bernoulli(ps); // note: not generative - this is a simplified distribution to make the log likelihood computations work quickly in Stan
}
"""

def create_pop_model(grp : str, n_trials : int = 50, fold_turns : bool = False, last : bool = False, use_turns : bool = True) -> populationOneBackRewardContextModel:
    subs = dl.get_subjects(grp);
    Y = np.ndarray((n_trials, len(subs)), dtype=int)
    R = np.ndarray((n_trials, len(subs)), dtype=int)

    for ii,sub in enumerate(subs):
        seqs,rs = dl.load_raw_with_reward_phase2(sub, use_turns=use_turns)
        seqs = np.concatenate(seqs);
        rs = np.concatenate(rs);
        if(last):
            Y[:,ii] = seqs[-n_trials:];
            R[:,ii] = rs[-n_trials:];
        else:
            Y[:,ii] = seqs[:n_trials];
            R[:,ii] = rs[:n_trials];

    if(fold_turns and use_turns):
        Y[Y == 2] = 0
        base_measure = np.array([2, 1]);
    elif(use_turns):
        base_measure = np.ones((3))
    else:
        base_measure = np.ones((4))

    return populationOneBackRewardContextModel(Y, R, base_measure=base_measure)

def create_indiv_model(sub : str, n_trials : int = 50, fold_turns : bool = False, last : bool = False, use_turns : bool = True) -> populationOneBackRewardContextModel:


    seqs,rs = dl.load_raw_with_reward_phase2(sub, use_turns=use_turns)
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

    if(fold_turns and use_turns):
        Y[Y == 2] = 0
        base_measure = np.array([2, 1]);
    elif(use_turns):
        base_measure = np.ones((3))
    else:
        base_measure = np.ones((4))

    return populationOneBackRewardContextModel(Y, R, base_measure = base_measure)