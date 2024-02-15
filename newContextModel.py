import numpy as np
import numpy.typing as npt
import stan, stan.model, stan.fit
import arviz as az
import pandas as pd
from hddCRP import dataLoader as dl


class populationTwoBackContextModel:
    def __init__(self, Y : npt.NDArray[np.int_]):
        assert Y.ndim == 2, "Y must be 2-D"
        assert Y.shape[1] > 1, "currently, needs to be population model"
        self.Y_raw_ = Y
        self.symbols_,self.Y_ = np.unique(Y, return_inverse=True);
    
        self.prior_context_similarity_depth_1_alpha_ = 1.0
        self.prior_context_similarity_depth_1_beta_ = 1.0
        self.prior_context_similarity_depth_2_alpha_ = 1.0
        self.prior_context_similarity_depth_2_beta_ = 1.0
        self.prior_subject_similarity_alpha_ = 1.0
        self.prior_subject_similarity_beta_ = 1.0

        self.prior_alpha_shape_ = 2.0
        self.prior_alpha_scale_ = 2.0

        self.prior_timeconstant_within_session_shape_ = 2.0
        self.prior_timeconstant_within_session_scale_ = 20.0

    def build(self, random_seed : int) -> stan.model.Model:
        self.posterior = stan.build(self.model, data=self.data_dict(), random_seed=int(random_seed))

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
    
    def data_dict(self) -> dict:
        
        data = {"N" : self.N, "M" : self.M, "K" : self.K,
                "Y" : self.Y_,
                "prior_alpha_shape" : self.prior_alpha_shape_,
                "prior_alpha_scale" : self.prior_alpha_scale_,
                "prior_timeconstant_within_session_shape" : self.prior_timeconstant_within_session_shape_,
                "prior_timeconstant_within_session_scale" : self.prior_timeconstant_within_session_scale_,
                "prior_context_similarity_depth_1_alpha" : self.prior_context_similarity_depth_1_alpha_,
                "prior_context_similarity_depth_1_beta" : self.prior_context_similarity_depth_1_beta_,
                "prior_context_similarity_depth_2_alpha" : self.prior_context_similarity_depth_2_alpha_,
                "prior_context_similarity_depth_2_beta" : self.prior_context_similarity_depth_2_beta_,
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

    real prior_alpha_shape;
    real prior_alpha_scale;
    
    real prior_timeconstant_within_session_shape;
    real prior_timeconstant_within_session_scale;
    

    real prior_context_similarity_depth_1_alpha;
    real prior_context_similarity_depth_1_beta;

    real prior_context_similarity_depth_2_alpha;
    real prior_context_similarity_depth_2_beta;
    
    real prior_subject_similarity_alpha;
    real prior_subject_similarity_beta;
}

transformed data {
    matrix[N*K,N*K] valid    = rep_matrix(0, N*K, N*K);
    matrix[N*K,N*K] is_same  = rep_matrix(0, N*K, N*K);
    matrix[N*K,N*K] delta_ts = rep_matrix(0, N*K, N*K);

    matrix[N*K,N*K] different_subjects = rep_matrix(0, N*K, N*K);

    matrix[N*K,*KN] context_depth_1 = rep_matrix(0, N, N);
    array[M] matrix[N*K,N*K] context_depth_2;

    for(mm in 1:M) {
        context_depth_2[mm] = rep_matrix(0, N, N);
    }

    for(jj in 1:K) {
        int oo_a = (jj-1)*N;
        for(kk in 1:K) {
            int oo_b = (kk-1)*N;
            for(aa in 1:N) {
                for(bb in 1:(aa-1)) {
                    delta_ts[oo_a + aa, oo_b+bb] = aa-bb;
                    valid[oo_a + aa, oo_b+bb] = 1;
                    if(jj != kk) {
                        different_subjects[oo_a + aa, oo_b+bb] = 1;
                    }
                    if(Y[aa,jj] == Y[bb,kk]) {
                        is_same[oo_a + aa, oo_b + bb] = 1;
                    }

                    if(aa > 1 && bb > 1) {
                        Y_c = Y[aa-1,jj];
                        if(Y_c != Y[bb-1,kk]) {
                            context_depth_1[oo_a + aa, oo_b+bb] = 1;
                        }
                        else {
                            if(aa > 2 && bb > 2) {
                                if(Y[aa-2,kk] != Y[bb-2,kk]) {
                                    context_depth_2[Y_c][oo_a + aa, oo_b+bb] = 1;
                                }
                            }
                        }
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
    matrix[N*K,N*K] weights_same_obs;
    matrix[N*K,N*K] weights_all_obs;
}

parameters {
    
    real<lower=0> alpha;   
    real<lower=0> timeconstant_within_session_A;
    real<lower=0,upper=1> subject_similarity;
    real<lower=0,upper=1> context_similarity_depth_1;
    vector<lower=0,upper=1>[M] context_similarity_depth_2;
}
model {

    alpha                          ~ gamma(prior_alpha_shape,                         prior_alpha_scale_inv);
    timeconstant_within_session_A  ~ gamma(prior_timeconstant_within_session_shape,   prior_timeconstant_within_session_scale_inv);
    context_similarity_depth_1     ~ beta(prior_context_similarity_depth_1_alpha,   prior_context_similarity_depth_1_beta);
    context_similarity_depth_2     ~ beta(prior_context_similarity_depth_2_alpha,   prior_context_similarity_depth_2_beta);
    subject_similarity     ~ beta(prior_subject_similarity_alpha,   prior_subject_similarity_beta);


    vector[N*K] ps;
    weights_all_obs   =  -(delta_ts./timeconstant_within_session_A)
                                                   + (log1m(subject_similarity)  * different_subjects)
                                                   + (log1m(context_similarity_depth_1)  * is_different_context_1);
    for(mm in 1:M) {
        weights_all_obs += log1m(context_similarity_depth_2[mm])  * context_depth_2[mm];
    }
    weights_all_obs   = valid .* exp(weights_all_obs);

    weights_same_obs  = is_same .* weights_all_obs;

    // probability of drawing the observed observations given their pasts
    ps =  ((weights_same_obs * vs) + (alpha/M)) ./  ((weights_all_obs * vs) + (alpha));
    
    1 ~ bernoulli(ps); // note: not generative - this is a simplified distribution to make the log likelihood computations work quickly in Stan
}
"""



def create_pop_model(grp : str, n_trials : int = 50, fold_turns : bool = False) -> populationTwoBackContextModel:
    subs = dl.get_subjects(grp);
    Y = np.ndarray((n_trials, len(subs)), dtype=int)

    for ii,sub in enumerate(subs):
        seqs = np.concatenate( dl.get_phase2(sub));
        Y[:,ii] = seqs[:n_trials];
    if(fold_turns):
        Y[Y == 2] = 0
    return populationTwoBackContextModel(Y)