
## ============================================================================================================================================================
model_individual_session_context_0_repeat_0 = """
data {
    int N; // Number of data points
    int M; // number of possible observations
    array[N] int Y;
    matrix[N,N] is_same_1_back;

    real prior_alpha_shape;
    real prior_alpha_scale;

    real prior_timeconstant_within_session_shape;
    real prior_timeconstant_within_session_scale;
}
transformed data {
    // variables to turn main computation in matrix operations
    matrix[N,N] is_same_observation;
    matrix[N,N] is_prev_observation;
    matrix[N,N] deltas;
    for (aa in 1:N) {
        for (bb in 1:N) {
            if(bb < aa) {
                deltas[aa,bb] = aa-bb;
                is_prev_observation[aa,bb] = 1;
            }
            else {
                deltas[aa,bb] = 1;
                is_prev_observation[aa,bb] = 0;
            }
            if(bb < aa && Y[aa] == Y[bb]) {
                is_same_observation[aa,bb] = 1;
            }
            else {
                is_same_observation[aa,bb] = 0;
            }
        }
    }
    
    vector[N] vs = rep_vector(1.0,N) ;
    array[N] int vi ;
    for (aa in 1:N) {
        vi[aa] = 1;
    }

    // prior parameter transformations for computations
    real prior_alpha_scale_log = log(prior_alpha_scale);
    real prior_alpha_scale_inv = 1.0/prior_alpha_scale;

    real prior_timeconstant_within_session_scale_log = log(prior_timeconstant_within_session_scale);
    real prior_timeconstant_within_session_scale_inv = 1.0/prior_timeconstant_within_session_scale;

}
parameters {
    real log_alpha_n;   
    real log_timeconstant_within_session_n;      
}
transformed parameters { 
    real log_alpha;   
    real log_timeconstant_within_session;      
    real log_repeat_bias_1_back; 

    log_alpha                       = log_alpha_n                       + prior_alpha_scale_log;
    log_timeconstant_within_session = log_timeconstant_within_session_n + prior_timeconstant_within_session_scale_log;

    real<lower=0> alpha;
    real<lower=0> timeconstant_within_session;   

    alpha                       = exp(log_alpha);
    timeconstant_within_session = exp(log_timeconstant_within_session); 
}
model {
    matrix[N,N] weights_same_obs;
    matrix[N,N] weights_all_obs;
    vector[N] ps;
    vector[N] aa;
    vector[N] BaseMeasure;
    real BaseMeasure;
    
    alpha                       ~ gamma(prior_alpha_shape,                       prior_alpha_scale_inv);
    timeconstant_within_session ~ gamma(prior_timeconstant_within_session_shape, prior_timeconstant_within_session_scale_inv);

    BaseMeasure = 1.0/M;

    weights_all_obs   = is_prev_observation .* exp(-deltas/timeconstant_within_session );
    weights_same_obs  = is_same_observation .* weights_all_obs[:,1:N];

    // probability of drawing the observed observations given their pasts
    ps =  ((weights_same_obs * vs) + (alpha*BaseMeasure)) ./  ((weights_all_obs * vs) + (alpha));

    vi ~ bernoulli(ps); // note: not generative - this is a simplified distribution to make the log likelihood computations work quickly in Stan
}"""

## ============================================================================================================================================================
model_individual_session_context_1_repeat_0 = """
data {
    int N; // Number of data points
    int M; // number of possible observations
    array[N] int Y;
    matrix[N,N] is_same_context_1;

    real prior_alpha_shape;
    real prior_alpha_scale;

    real prior_timeconstant_within_session_shape;
    real prior_timeconstant_within_session_scale;

    real prior_context_similarity_depth_1_alpha;
    real prior_context_similarity_depth_1_beta;
}
transformed data {
    // variables to turn main computation in matrix operations
    matrix[N,N] is_same_observation;
    matrix[N,N] is_prev_observation;
    matrix[N,N] is_different_context_1;
    matrix[N,N] deltas;
    for (aa in 1:N) {
        for (bb in 1:N) {
            if(bb < aa) {
                deltas[aa,bb] = aa-bb;
                is_prev_observation[aa,bb] = 1;
            }
            else {
                deltas[aa,bb] = 1;
                is_prev_observation[aa,bb] = 0;
            }
            if(bb < aa && Y[aa] == Y[bb]) {
                is_same_observation[aa,bb] = 1;
            }
            else {
                is_same_observation[aa,bb] = 0;
            }
            
            if(bb < aa && is_same_context_1[aa,bb] <= 0) {
                is_different_context_1[aa,bb] = 1;
            }
            else {
                is_different_context_1[aa,bb] = 0;
            }
        }
    }
    
    vector[N] vs = rep_vector(1.0,N) ;
    array[N] int vi ;
    for (aa in 1:N) {
        vi[aa] = 1;
    }

    // prior parameter transformations for computations
    real prior_alpha_scale_log = log(prior_alpha_scale);
    real prior_alpha_scale_inv = 1.0/prior_alpha_scale;

    real prior_timeconstant_within_session_scale_log = log(prior_timeconstant_within_session_scale);
    real prior_timeconstant_within_session_scale_inv = 1.0/prior_timeconstant_within_session_scale;

    real prior_repeat_bias_1_back_scale     = 1.0/prior_repeat_bias_1_back_shape;
    real prior_repeat_bias_1_back_scale_inv = prior_repeat_bias_1_back_shape;
    real prior_repeat_bias_1_back_scale_log = log(prior_repeat_bias_1_back_scale);
}
parameters {
    real log_alpha_n;   
    real log_timeconstant_within_session_n;   
    
    //real logit_context_similarity_depth_1;  
    real<upper=0> log_context_similarity_depth_1;     
}
transformed parameters { 
    real log_alpha;   
    real log_timeconstant_within_session;  

    log_alpha                       = log_alpha_n                       + prior_alpha_scale_log;
    log_timeconstant_within_session = log_timeconstant_within_session_n + prior_timeconstant_within_session_scale_log;

    real<lower=0> alpha;
    real<lower=0> timeconstant_within_session; 

    alpha                       = exp(log_alpha);
    timeconstant_within_session = exp(log_timeconstant_within_session); 

    real<lower=0,upper=1> context_similarity_depth_1; 
    context_similarity_depth_1     = exp(log_context_similarity_depth_1);
}
model {
    matrix[N,N] weights_same_obs;
    matrix[N,N] weights_all_obs;
    vector[N] ps;
    vector[N] aa;
    real BaseMeasure;
    
    alpha                       ~ gamma(prior_alpha_shape,                       prior_alpha_scale_inv);
    timeconstant_within_session ~ gamma(prior_timeconstant_within_session_shape, prior_timeconstant_within_session_scale_inv);

    context_similarity_depth_1  ~ beta(prior_context_similarity_depth_1_alpha,   prior_context_similarity_depth_1_beta);
    
    BaseMeasure = 1.0/M;

    weights_all_obs   = is_prev_observation .* exp(-deltas/timeconstant_within_session 
                                                   + log_context_similarity_depth_1   * is_different_context_1);
    weights_same_obs  = is_same_observation .* weights_all_obs[:,1:N];

    // probability of drawing the observed observations given their pasts
    ps =  ((weights_same_obs * vs) + (alpha*BaseMeasure)) ./  ((weights_all_obs * vs) + (alpha));

    vi ~ bernoulli(ps); // note: not generative - this is a simplified distribution to make the log likelihood computations work quickly in Stan
}"""

## ============================================================================================================================================================
model_individual_session_context_2_repeat_0 = """
data {
    int N; // Number of data points
    int M; // number of possible observations
    array[N] int Y;
    matrix[N,N] is_same_context_1;
    matrix[N,N] is_same_context_2;

    real prior_alpha_shape;
    real prior_alpha_scale;

    real prior_timeconstant_within_session_shape;
    real prior_timeconstant_within_session_scale;

    real prior_context_similarity_depth_1_alpha;
    real prior_context_similarity_depth_1_beta;
    real prior_context_similarity_depth_2_alpha;
    real prior_context_similarity_depth_2_beta;
}
transformed data {
    // variables to turn main computation in matrix operations
    matrix[N,N] is_same_observation;
    matrix[N,N] is_prev_observation;
    matrix[N,N] is_different_context_1;
    matrix[N,N] is_different_context_2;
    matrix[N,N] deltas;
    for (aa in 1:N) {
        for (bb in 1:N) {
            if(bb < aa) {
                deltas[aa,bb] = aa-bb;
                is_prev_observation[aa,bb] = 1;
            }
            else {
                deltas[aa,bb] = 1;
                is_prev_observation[aa,bb] = 0;
            }
            if(bb < aa && Y[aa] == Y[bb]) {
                is_same_observation[aa,bb] = 1;
            }
            else {
                is_same_observation[aa,bb] = 0;
            }
            
            if(bb < aa && is_same_context_1[aa,bb] <= 0) {
                is_different_context_1[aa,bb] = 1;
            }
            else {
                is_different_context_1[aa,bb] = 0;
            }
            if(bb < aa && is_same_context_2[aa,bb] <= 0) {
                is_different_context_2[aa,bb] = 1;
            }
            else {
                is_different_context_2[aa,bb] = 0;
            }
        }
    }
    
    vector[N] vs = rep_vector(1.0,N) ;
    array[N] int vi ;
    for (aa in 1:N) {
        vi[aa] = 1;
    }

    // prior parameter transformations for computations
    real prior_alpha_scale_log = log(prior_alpha_scale);
    real prior_alpha_scale_inv = 1.0/prior_alpha_scale;

    real prior_timeconstant_within_session_scale_log = log(prior_timeconstant_within_session_scale);
    real prior_timeconstant_within_session_scale_inv = 1.0/prior_timeconstant_within_session_scale;
}
parameters {
    real log_alpha_n;   
    real log_timeconstant_within_session_n;   
    
    real<upper=0> log_context_similarity_depth_1;    
    real<upper=0> log_context_similarity_depth_2;    
}
transformed parameters { 
    real log_alpha;   
    real log_timeconstant_within_session;      
    real log_repeat_bias_1_back; 

    log_alpha                       = log_alpha_n                       + prior_alpha_scale_log;
    log_timeconstant_within_session = log_timeconstant_within_session_n + prior_timeconstant_within_session_scale_log;

    real<lower=0> alpha;
    real<lower=0> timeconstant_within_session;    

    alpha                       = exp(log_alpha);
    timeconstant_within_session = exp(log_timeconstant_within_session); 

    real<lower=0,upper=1> context_similarity_depth_1; 
    real<lower=0,upper=1> context_similarity_depth_2; 
    
    context_similarity_depth_1       = exp(log_context_similarity_depth_1);
    context_similarity_depth_2       = exp(log_context_similarity_depth_2);
}
model {
    matrix[N,N] weights_same_obs;
    matrix[N,N] weights_all_obs;
    vector[N] ps;
    vector[N] aa;
    vector[N] BaseMeasure;
    real BaseMeasure;
    
    alpha                       ~ gamma(prior_alpha_shape,                       prior_alpha_scale_inv);
    timeconstant_within_session ~ gamma(prior_timeconstant_within_session_shape, prior_timeconstant_within_session_scale_inv);

    context_similarity_depth_1  ~ beta(prior_context_similarity_depth_1_alpha,   prior_context_similarity_depth_1_beta);
    context_similarity_depth_2  ~ beta(prior_context_similarity_depth_2_alpha,   prior_context_similarity_depth_2_beta);
    
    BaseMeasure = 1.0/M;

    weights_all_obs   = is_prev_observation .* exp(-deltas/timeconstant_within_session 
                                                   + log_context_similarity_depth_1   * is_different_context_1
                                                   + log_context_similarity_depth_2   * is_different_context_2);
    weights_same_obs  = is_same_observation .* weights_all_obs[:,1:N];

    // probability of drawing the observed observations given their pasts
    ps =  ((weights_same_obs * vs) + (alpha*BaseMeasure)) ./  ((weights_all_obs * vs) + (alpha));

    vi ~ bernoulli(ps); // note: not generative - this is a simplified distribution to make the log likelihood computations work quickly in Stan
}"""
## ============================================================================================================================================================
model_individual_session_context_0_repeat_1 = """
data {
    int N; // Number of data points
    int M; // number of possible observations
    array[N] int Y;
    matrix[N,N] is_same_1_back;

    real prior_alpha_shape;
    real prior_alpha_scale;

    real prior_timeconstant_within_session_shape;
    real prior_timeconstant_within_session_scale;

    real prior_repeat_bias_1_back_shape;
    real prior_repeat_bias_1_back_scale;
}
transformed data {
    // variables to turn main computation in matrix operations
    matrix[N,N] is_same_observation;
    matrix[N,N] is_prev_observation;
    matrix[N,N] deltas;
    for (aa in 1:N) {
        for (bb in 1:N) {
            if(bb < aa) {
                deltas[aa,bb] = aa-bb;
                is_prev_observation[aa,bb] = 1;
            }
            else {
                deltas[aa,bb] = 1;
                is_prev_observation[aa,bb] = 0;
            }
            if(bb < aa && Y[aa] == Y[bb]) {
                is_same_observation[aa,bb] = 1;
            }
            else {
                is_same_observation[aa,bb] = 0;
            }
        }
    }

    vector[N] Y_is_same_as_one_back = rep_vector(0, N);
    for (aa in 1:N) {
        if(is_same_1_back[aa,aa] > 0) {
            Y_is_same_as_one_back[aa] = 1;
        }
    }
    
    vector[N] vs = rep_vector(1.0,N) ;
    array[N] int vi ;
    for (aa in 1:N) {
        vi[aa] = 1;
    }

    // prior parameter transformations for computations
    real prior_alpha_scale_log = log(prior_alpha_scale);
    real prior_alpha_scale_inv = 1.0/prior_alpha_scale;

    real prior_timeconstant_within_session_scale_log = log(prior_timeconstant_within_session_scale);
    real prior_timeconstant_within_session_scale_inv = 1.0/prior_timeconstant_within_session_scale;

    real prior_repeat_bias_1_back_scale     = 1.0/prior_repeat_bias_1_back_shape;
    real prior_repeat_bias_1_back_scale_inv = prior_repeat_bias_1_back_shape;
    real prior_repeat_bias_1_back_scale_log = log(prior_repeat_bias_1_back_scale);
}
parameters {
    real log_alpha_n;   
    real log_timeconstant_within_session_n;      
    real log_repeat_bias_1_back_n;   
}
transformed parameters { 
    real log_alpha;   
    real log_timeconstant_within_session;      
    real log_repeat_bias_1_back; 

    log_alpha                       = log_alpha_n                       + prior_alpha_scale_log;
    log_timeconstant_within_session = log_timeconstant_within_session_n + prior_timeconstant_within_session_scale_log;
    log_repeat_bias_1_back          = log_repeat_bias_1_back_n          + prior_repeat_bias_1_back_scale_log;

    real<lower=0> alpha;
    real<lower=0> timeconstant_within_session;         
    real<lower=0> repeat_bias_1_back;   

    alpha                       = exp(log_alpha);
    timeconstant_within_session = exp(log_timeconstant_within_session); 
    repeat_bias_1_back          = exp(log_repeat_bias_1_back);
}
model {
    matrix[N,N] weights_same_obs;
    matrix[N,N] weights_all_obs;
    vector[N] ps;
    vector[N] aa;
    vector[N] BaseMeasure;
    // real BaseMeasure;
    
    alpha                       ~ gamma(prior_alpha_shape,                       prior_alpha_scale_inv);
    timeconstant_within_session ~ gamma(prior_timeconstant_within_session_shape, prior_timeconstant_within_session_scale_inv);
    repeat_bias_1_back          ~ gamma(prior_repeat_bias_1_back_shape,          prior_repeat_bias_1_back_scale_inv);
    
    BaseMeasure = ((repeat_bias_1_back-1.0) * Y_is_same_as_one_back + 1.0) / (repeat_bias_1_back + (M-1.0));
    //BaseMeasure = 1.0/M;

    weights_all_obs   = is_prev_observation .* exp(-deltas/timeconstant_within_session 
                                                   + log_repeat_bias_1_back           * is_same_1_back);
    //weights_all_obs = is_prev_observation .* exp(-deltas/timeconstant_within_session);
    weights_same_obs  = is_same_observation .* weights_all_obs[:,1:N];

    // probability of drawing the observed observations given their pasts
    ps =  ((weights_same_obs * vs) + (alpha*BaseMeasure)) ./  ((weights_all_obs * vs) + (alpha));

    vi ~ bernoulli(ps); // note: not generative - this is a simplified distribution to make the log likelihood computations work quickly in Stan
}"""

## ============================================================================================================================================================
model_individual_session_context_1_repeat_1 = """
data {
    int N; // Number of data points
    int M; // number of possible observations
    array[N] int Y;
    matrix[N,N] is_same_context_1;
    matrix[N,N] is_same_1_back;

    real prior_alpha_shape;
    real prior_alpha_scale;

    real prior_timeconstant_within_session_shape;
    real prior_timeconstant_within_session_scale;

    real prior_repeat_bias_1_back_shape;
    real prior_repeat_bias_1_back_scale;

    real prior_context_similarity_depth_1_alpha;
    real prior_context_similarity_depth_1_beta;
}
transformed data {
    // variables to turn main computation in matrix operations
    matrix[N,N] is_same_observation;
    matrix[N,N] is_prev_observation;
    matrix[N,N] is_different_context_1;
    matrix[N,N] deltas;
    for (aa in 1:N) {
        for (bb in 1:N) {
            if(bb < aa) {
                deltas[aa,bb] = aa-bb;
                is_prev_observation[aa,bb] = 1;
            }
            else {
                deltas[aa,bb] = 1;
                is_prev_observation[aa,bb] = 0;
            }
            if(bb < aa && Y[aa] == Y[bb]) {
                is_same_observation[aa,bb] = 1;
            }
            else {
                is_same_observation[aa,bb] = 0;
            }
            
            if(bb < aa && is_same_context_1[aa,bb] <= 0) {
                is_different_context_1[aa,bb] = 1;
            }
            else {
                is_different_context_1[aa,bb] = 0;
            }
        }
    }

    vector[N] Y_is_same_as_one_back = rep_vector(0, N);
    for (aa in 1:N) {
        if(is_same_1_back[aa,aa] > 0) {
            Y_is_same_as_one_back[aa] = 1;
        }
    }
    
    vector[N] vs = rep_vector(1.0,N) ;
    array[N] int vi ;
    for (aa in 1:N) {
        vi[aa] = 1;
    }

    // prior parameter transformations for computations
    real prior_alpha_scale_log = log(prior_alpha_scale);
    real prior_alpha_scale_inv = 1.0/prior_alpha_scale;

    real prior_timeconstant_within_session_scale_log = log(prior_timeconstant_within_session_scale);
    real prior_timeconstant_within_session_scale_inv = 1.0/prior_timeconstant_within_session_scale;

    real prior_repeat_bias_1_back_scale     = 1.0/prior_repeat_bias_1_back_shape;
    real prior_repeat_bias_1_back_scale_inv = prior_repeat_bias_1_back_shape;
    real prior_repeat_bias_1_back_scale_log = log(prior_repeat_bias_1_back_scale);
}
parameters {
    real log_alpha_n;   
    real log_timeconstant_within_session_n;      
    real log_repeat_bias_1_back_n;   
    
    //real logit_context_similarity_depth_1;  
    real<upper=0> log_context_similarity_depth_1;     
}
transformed parameters { 
    real log_alpha;   
    real log_timeconstant_within_session;      
    real log_repeat_bias_1_back; 

    log_alpha                       = log_alpha_n                       + prior_alpha_scale_log;
    log_timeconstant_within_session = log_timeconstant_within_session_n + prior_timeconstant_within_session_scale_log;
    log_repeat_bias_1_back          = log_repeat_bias_1_back_n          + prior_repeat_bias_1_back_scale_log;

    real<lower=0> alpha;
    real<lower=0> timeconstant_within_session;         
    real<lower=0> repeat_bias_1_back;   

    alpha                       = exp(log_alpha);
    timeconstant_within_session = exp(log_timeconstant_within_session); 
    repeat_bias_1_back          = exp(log_repeat_bias_1_back);

    real<lower=0,upper=1> context_similarity_depth_1; 
    //real<upper=0>         log_context_similarity_depth_1;

    //context_similarity_depth_1     = inv_logit(logit_context_similarity_depth_1);
    //log_context_similarity_depth_1 = log(context_similarity_depth_1);
    context_similarity_depth_1     = exp(log_context_similarity_depth_1);
}
model {
    matrix[N,N] weights_same_obs;
    matrix[N,N] weights_all_obs;
    vector[N] ps;
    vector[N] aa;
    vector[N] BaseMeasure;
    // real BaseMeasure;
    
    alpha                       ~ gamma(prior_alpha_shape,                       prior_alpha_scale_inv);
    timeconstant_within_session ~ gamma(prior_timeconstant_within_session_shape, prior_timeconstant_within_session_scale_inv);
    repeat_bias_1_back          ~ gamma(prior_repeat_bias_1_back_shape,          prior_repeat_bias_1_back_scale_inv);

    context_similarity_depth_1  ~ beta(prior_context_similarity_depth_1_alpha,   prior_context_similarity_depth_1_beta);
    
    BaseMeasure = ((repeat_bias_1_back-1.0) * Y_is_same_as_one_back + 1.0) / (repeat_bias_1_back + (M-1.0));
    //BaseMeasure = 1.0/M;

    weights_all_obs   = is_prev_observation .* exp(-deltas/timeconstant_within_session 
                                                   + log_context_similarity_depth_1   * is_different_context_1
                                                   + log_repeat_bias_1_back           * is_same_1_back);
    //weights_all_obs   = is_prev_observation .* exp(-deltas/timeconstant_within_session);
    weights_same_obs  = is_same_observation .* weights_all_obs[:,1:N];

    // probability of drawing the observed observations given their pasts
    ps =  ((weights_same_obs * vs) + (alpha*BaseMeasure)) ./  ((weights_all_obs * vs) + (alpha));

    vi ~ bernoulli(ps); // note: not generative - this is a simplified distribution to make the log likelihood computations work quickly in Stan
}"""

## ============================================================================================================================================================
model_individual_session_context_2_repeat_1 = """
data {
    int N; // Number of data points
    int M; // number of possible observations
    array[N] int Y;
    matrix[N,N] is_same_context_1;
    matrix[N,N] is_same_context_2;
    matrix[N,N] is_same_1_back;

    real prior_alpha_shape;
    real prior_alpha_scale;

    real prior_timeconstant_within_session_shape;
    real prior_timeconstant_within_session_scale;

    real prior_repeat_bias_1_back_shape;
    real prior_repeat_bias_1_back_scale;

    real prior_context_similarity_depth_1_alpha;
    real prior_context_similarity_depth_1_beta;
    real prior_context_similarity_depth_2_alpha;
    real prior_context_similarity_depth_2_beta;
}
transformed data {
    // variables to turn main computation in matrix operations
    matrix[N,N] is_same_observation;
    matrix[N,N] is_prev_observation;
    matrix[N,N] is_different_context_1;
    matrix[N,N] is_different_context_2;
    matrix[N,N] deltas;
    for (aa in 1:N) {
        for (bb in 1:N) {
            if(bb < aa) {
                deltas[aa,bb] = aa-bb;
                is_prev_observation[aa,bb] = 1;
            }
            else {
                deltas[aa,bb] = 1;
                is_prev_observation[aa,bb] = 0;
            }
            if(bb < aa && Y[aa] == Y[bb]) {
                is_same_observation[aa,bb] = 1;
            }
            else {
                is_same_observation[aa,bb] = 0;
            }
            
            if(bb < aa && is_same_context_1[aa,bb] <= 0) {
                is_different_context_1[aa,bb] = 1;
            }
            else {
                is_different_context_1[aa,bb] = 0;
            }
            if(bb < aa && is_same_context_2[aa,bb] <= 0) {
                is_different_context_2[aa,bb] = 1;
            }
            else {
                is_different_context_2[aa,bb] = 0;
            }
        }
    }

    vector[N] Y_is_same_as_one_back = rep_vector(0, N);
    for (aa in 1:N) {
        if(is_same_1_back[aa,aa] > 0) {
            Y_is_same_as_one_back[aa] = 1;
        }
    }

    vector[N] vs = rep_vector(1.0,N) ;
    array[N] int vi ;
    for (aa in 1:N) {
        vi[aa] = 1;
    }

    // prior parameter transformations for computations
    real prior_alpha_scale_log = log(prior_alpha_scale);
    real prior_alpha_scale_inv = 1.0/prior_alpha_scale;

    real prior_timeconstant_within_session_scale_log = log(prior_timeconstant_within_session_scale);
    real prior_timeconstant_within_session_scale_inv = 1.0/prior_timeconstant_within_session_scale;

    real prior_repeat_bias_1_back_scale     = 1.0/prior_repeat_bias_1_back_shape;
    real prior_repeat_bias_1_back_scale_inv = prior_repeat_bias_1_back_shape;
    real prior_repeat_bias_1_back_scale_log = log(prior_repeat_bias_1_back_scale);
}
parameters {
    real log_alpha_n;   
    real log_timeconstant_within_session_n;      
    real log_repeat_bias_1_back_n;   
    
    //real logit_context_similarity_depth_1;  
    real<upper=0> log_context_similarity_depth_1;    
    real<upper=0> log_context_similarity_depth_2;    
}
transformed parameters { 
    real log_alpha;   
    real log_timeconstant_within_session;      
    real log_repeat_bias_1_back; 

    log_alpha                       = log_alpha_n                       + prior_alpha_scale_log;
    log_timeconstant_within_session = log_timeconstant_within_session_n + prior_timeconstant_within_session_scale_log;
    log_repeat_bias_1_back          = log_repeat_bias_1_back_n          + prior_repeat_bias_1_back_scale_log;

    real<lower=0> alpha;
    real<lower=0> timeconstant_within_session;         
    real<lower=0> repeat_bias_1_back;   

    alpha                       = exp(log_alpha);
    timeconstant_within_session = exp(log_timeconstant_within_session); 
    repeat_bias_1_back          = exp(log_repeat_bias_1_back);

    real<lower=0,upper=1> context_similarity_depth_1; 
    real<lower=0,upper=1> context_similarity_depth_2; 

    context_similarity_depth_1       = exp(log_context_similarity_depth_1);
    context_similarity_depth_2       = exp(log_context_similarity_depth_2);
}
model {
    matrix[N,N] weights_same_obs;
    matrix[N,N] weights_all_obs;
    vector[N] ps;
    vector[N] aa;
    vector[N] BaseMeasure;
    // real BaseMeasure;
    
    alpha                       ~ gamma(prior_alpha_shape,                       prior_alpha_scale_inv);
    timeconstant_within_session ~ gamma(prior_timeconstant_within_session_shape, prior_timeconstant_within_session_scale_inv);
    repeat_bias_1_back          ~ gamma(prior_repeat_bias_1_back_shape,          prior_repeat_bias_1_back_scale_inv);

    context_similarity_depth_1  ~ beta(prior_context_similarity_depth_1_alpha,   prior_context_similarity_depth_1_beta);
    context_similarity_depth_2  ~ beta(prior_context_similarity_depth_2_alpha,   prior_context_similarity_depth_2_beta);
    
    BaseMeasure = ((repeat_bias_1_back-1.0) * Y_is_same_as_one_back + 1.0) / (repeat_bias_1_back + (M-1.0));
    //BaseMeasure = 1.0/M;

    weights_all_obs   = is_prev_observation .* exp(-deltas/timeconstant_within_session 
                                                   + log_context_similarity_depth_1   * is_different_context_1
                                                   + log_context_similarity_depth_2   * is_different_context_2
                                                   + log_repeat_bias_1_back           * is_same_1_back);
    //weights_all_obs   = is_prev_observation .* exp(-deltas/timeconstant_within_session);
    weights_same_obs  = is_same_observation .* weights_all_obs[:,1:N];

    // probability of drawing the observed observations given their pasts
    ps =  ((weights_same_obs * vs) + (alpha*BaseMeasure)) ./  ((weights_all_obs * vs) + (alpha));

    vi ~ bernoulli(ps); // note: not generative - this is a simplified distribution to make the log likelihood computations work quickly in Stan
}"""