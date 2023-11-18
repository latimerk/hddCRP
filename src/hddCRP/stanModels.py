from __future__ import annotations

def generate_stan_code_individual(within_session_timeconstants : list,
                                  session_interaction_types : list,
                                  context_depth : int,
                                  same_nback_depth : int,
                                  repeat_bias_in_connection_weights : bool = False) -> str:
    context_depth = max(0, int(context_depth))
    same_nback_depth = max(0, int(same_nback_depth))

    within_session_timeconstants = [str(ss).replace(' ', '_') for ss in within_session_timeconstants]
    session_interaction_types = [str(ss).replace(' ', '_') for ss in session_interaction_types]

    include_within_session_timeconstants = len(within_session_timeconstants) > 0

    assert same_nback_depth <= 1, "can't generate model with nback biases greater than 1"
    lb = "{"
    rb = "}"

    data_block = """
data {
    int N; // Number of data points
    int M; // number of possible observations
    array[N] int Y;
    array[N] int session_id;
    array[N] int local_timeconstant_id;
    array[N] real local_time;


    int T; // max observation distance
    int K; // number of subjects that are stacked on top of each other
    array[K+1] int subject_start_idx; // first element should be 1, last element should be N+1

    real prior_alpha_shape;
    real prior_alpha_scale;
"""
    
    for ii in session_interaction_types:
         data_block += f"""    array[N,2] real session_time_{ii}; // column 1: projecting time, column 2: receiving time; if negative, doesn't do either
"""
    
    if(include_within_session_timeconstants):
        data_block += """
    real prior_timeconstant_within_session_shape;
    real prior_timeconstant_within_session_scale;
"""
    if(len(session_interaction_types) > 0):
        data_block += """    real prior_timeconstant_between_sessions_shape;
    real prior_timeconstant_between_sessions_scale;
"""

    for ii in range(1,context_depth+1):
        data_block += f"""    matrix[N,T] is_same_context_{ii};
"""
        data_block += f"""    real prior_context_similarity_depth_{ii}_alpha;
"""
        data_block += f"""    real prior_context_similarity_depth_{ii}_beta;
"""

    for ii in range(1,same_nback_depth+1):
        data_block += f"""    matrix[N,T] is_same_{ii}_back;
"""
        data_block += f"""    real prior_repeat_bias_{ii}_back_shape;
"""
    data_block += """}
"""

    transformed_data_block = """
transformed data {
    // variables to turn main computation in matrix operations
    vector[T] vs = rep_vector(1, T);

    matrix[N,T] is_same_observation = rep_matrix(0, N, T); // for numerator in CRP likelihood p(y_t | y_1:t-1)
    matrix[N,T] is_prev_observation = rep_matrix(0, N, T); // for denominator in CRP likelihood p(y_t | y_1:t-1)
    for (kk in 1:K) {
        int start_t = subject_start_idx[kk];
        int end_t   = subject_start_idx[kk+1];
        int t_c     = end_t - start_t;
        for (aa in 1:t_c) {
            int aa_c = aa + start_t - 1;
            for (bb in 1:t_c) {
                int bb_c = bb + start_t - 1;
                if(local_time[bb_c] < local_time[aa_c]) {
                    is_prev_observation[aa_c,bb] = 1;
                }

                if((is_prev_observation[aa_c,bb] > 0) && (Y[aa_c] == Y[bb_c])) {
                    is_same_observation[aa_c,bb] = 1;
                }
            }
        }
    }
    
"""
    for ii in within_session_timeconstants:
        transformed_data_block += f"""    matrix[N,T] deltas_{ii} = rep_matrix(0, N, T);
"""
        
    for ii in session_interaction_types:
        transformed_data_block += f"""    matrix[N,T] deltas_session_{ii} = rep_matrix(0, N, T);
"""
        
    transformed_data_block += """
    for (kk in 1:K) {
        int start_t = subject_start_idx[kk];
        int end_t   = subject_start_idx[kk+1];
        int t_c     = end_t - start_t;
        for (aa in 1:t_c) {
            int aa_c = aa + start_t - 1;
            for (bb in 1:t_c) {
                int bb_c = bb + start_t - 1;
"""

    for id_num_0, ii in enumerate(within_session_timeconstants):
        id_num = id_num_0 + 1;
        transformed_data_block += f"""
                if((is_prev_observation[aa_c,bb] > 0) && (local_timeconstant_id[aa_c] == {id_num}) && (local_timeconstant_id[bb_c] == {id_num}) && (session_id[aa_c] == session_id[bb_c])) {lb}
                    deltas_{ii}[aa_c,bb] = local_time[aa_c]-local_time[bb_c];
                {rb}
"""

    for ii in session_interaction_types:
        transformed_data_block += f"""
                if((is_prev_observation[aa_c,bb] > 0)&& (session_time_{ii}[aa_c,2] > 0) && (session_time_{ii}[bb_c,1] > 0)) {lb}
                    deltas_session_{ii}[aa_c,bb] = session_time_{ii}[aa_c,2]-session_time_{ii}[bb_c,1];
                {rb}
"""
    transformed_data_block += """
            }
        }
    }
"""
    for ii in range(1,context_depth+1):
        transformed_data_block += f"""
    matrix[N,T] is_different_context_{ii} = rep_matrix(0, N, T);
"""

    if(context_depth > 0):
        transformed_data_block += """
    for (kk in 1:K) {
        int start_t = subject_start_idx[kk];
        int end_t   = subject_start_idx[kk+1];
        int t_c     = end_t - start_t;
        for (aa in 1:t_c) {
            int aa_c = aa + start_t - 1;
            for (bb in 1:t_c) {
                int bb_c = bb + start_t - 1;
"""

        for ii in range(1,context_depth+1):
            transformed_data_block +=f"""
                if((is_prev_observation[aa_c,bb] > 0) && (is_same_context_{ii}[aa_c,bb] <= 0)) {lb}
                    is_different_context_{ii}[aa_c,bb] = 1;
                {rb}
"""

        transformed_data_block += """
            }
        }
    }
"""

    if(same_nback_depth > 0):
        for ii in range(1,same_nback_depth+1):
            transformed_data_block += f"""
    vector[N] Y_is_same_as_{ii}_back = rep_vector(0, N);
    vector[N] Y_is_not_start  = rep_vector(1, N);
    for (kk in 1:K) {lb}
        int start_t = subject_start_idx[kk];
        int end_t   = subject_start_idx[kk+1];
        int t_c     = end_t - start_t;
        Y_is_not_start[start_t] = 0;
        for (aa in 2:t_c) {lb}
            int aa_c = aa + start_t - 1;
            if(is_same_{ii}_back[aa_c,aa] > 0) {lb}
                Y_is_same_as_{ii}_back[aa_c] = 1;
            {rb}
        {rb}
    {rb}
"""

    transformed_data_block += """
    // prior parameter transformations for computations
    real prior_alpha_scale_inv = inv(prior_alpha_scale);
"""

    if(len(within_session_timeconstants) > 0):
        transformed_data_block += """
    real prior_timeconstant_within_session_scale_inv   = inv(prior_timeconstant_within_session_scale);
"""
    if(len(session_interaction_types) > 0):
        transformed_data_block += """
    real prior_timeconstant_between_sessions_scale_inv = inv(prior_timeconstant_between_sessions_scale);
"""

    for ii in range(1,same_nback_depth+1):
        transformed_data_block += f"""
    real prior_repeat_bias_{ii}_back_scale     = inv(prior_repeat_bias_1_back_shape);
    real prior_repeat_bias_{ii}_back_scale_inv = prior_repeat_bias_1_back_shape;
"""
    transformed_data_block += """}
"""

    parameters_block = """parameters {
    real<lower=0> alpha;   
"""
    for ii in within_session_timeconstants: 
        parameters_block += f"""    real<lower=0> timeconstant_within_session_{ii};
"""

    for ii in session_interaction_types: 
        parameters_block += f"""    real<lower=0> timeconstant_between_sessions_{ii};
"""

    for ii in range(1,same_nback_depth+1): 
        parameters_block += f"""    real<lower=0> repeat_bias_{ii}_back;
"""

    for ii in range(1,context_depth+1): 
        parameters_block += f"""    real<lower=0,upper=1> context_similarity_depth_{ii};
"""

    parameters_block += """}
"""

    
    model_block = """
model {
"""

    
    model_block += """
    alpha                         ~ gamma(prior_alpha_shape,                         prior_alpha_scale_inv);
"""
    
    for ii in within_session_timeconstants:
        model_block += f"""    timeconstant_within_session_{ii}   ~ gamma(prior_timeconstant_within_session_shape,   prior_timeconstant_within_session_scale_inv);
"""

    for ii in session_interaction_types:
        model_block += f"""    timeconstant_between_sessions_{ii} ~ gamma(prior_timeconstant_between_sessions_shape, prior_between_sessions_scale_inv);
"""


    for ii in range(1, same_nback_depth+1):
        model_block += f"""    repeat_bias_{ii}_back            ~ gamma(prior_repeat_bias_{ii}_back_shape,            prior_repeat_bias_{ii}_back_scale_inv);
"""

    for ii in range(1, context_depth+1):
        model_block += f"""    context_similarity_depth_{ii}  ~ beta(prior_context_similarity_depth_{ii}_alpha,   prior_context_similarity_depth_{ii}_beta);
"""

    prob_seating = ""   
    if(same_nback_depth > 0):
        depth = 1;
        prob_seating += f"""
    vector[N] BaseMeasure;
    BaseMeasure = (Y_is_same_as_1_back * (repeat_bias_1_back-1.0) + 1.0) ./ (Y_is_not_start * repeat_bias_1_back + (M-Y_is_not_start));
"""
    else:
        prob_seating += """
    real BaseMeasure;
    BaseMeasure = inv(M);
""" 
    prob_seating +=     f"""
    matrix[N,T] weights_same_obs;
    matrix[N,T] weights_all_obs;
    vector[N] ps;
    weights_all_obs   = is_prev_observation"""
    
    start_str = " .* exp("
    space_str = ""
    end_str = ""
    space_str_b = "                                                   "
    p_str = ""
    
    for ii in within_session_timeconstants:
        prob_seating += f"""{start_str}{space_str} -(deltas_{ii}./timeconstant_within_session_{ii})
"""
        space_str = space_str_b
        start_str = ""
        end_str = ")"
        p_str = "+"

    for ii in session_interaction_types:
        prob_seating += f"""{start_str}{space_str} -(deltas_session_{ii}./timeconstant_within_session_{ii})
"""
        space_str = space_str_b
        start_str = ""
        end_str = ")"
        p_str = "+"
        
    for ii in range(1, context_depth+1):
        prob_seating += f"""{start_str}{space_str}{p_str} (log1m(context_similarity_depth_{ii})  * is_different_context_{ii})
"""
        space_str = space_str_b
        start_str = ""
        end_str = ")"
        p_str = "+"

    if(repeat_bias_in_connection_weights):    
        for ii in range(1, same_nback_depth+1):
            prob_seating += f"""{start_str}{space_str}{p_str}  (log(context_repeat_bias_{ii}_back)   * is_same_{ii}_back)
    """
            space_str = space_str_b
            start_str = ""
            end_str = ")"
            p_str = "+"


    prob_seating += f"""{space_str}{end_str};
"""
        
    prob_seating += """
    weights_same_obs  = is_same_observation .* weights_all_obs;

    // probability of drawing the observed observations given their pasts
    ps =  ((weights_same_obs * vs) + (alpha*BaseMeasure)) ./  ((weights_all_obs * vs) + (alpha));
    """

    model_block += prob_seating;

    
    model_block += """
    1 ~ bernoulli(ps); // note: not generative - this is a simplified distribution to make the log likelihood computations work quickly in Stan
}
"""

    generated_quantities_block = f"""
generated quantities {lb}
    vector[K] log_likelihood;
    {lb}
        {prob_seating}
        for(kk in 1:K) {lb}
            int start_t = subject_start_idx[kk];
            int end_t   = subject_start_idx[kk+1]-1;
            log_likelihood[kk] = bernoulli_lpmf(1 | ps[start_t:end_t]);
        {rb}
    {rb}
{rb}
"""
    transformed_parameters_block = ""
    stan_model = data_block + transformed_data_block + parameters_block + transformed_parameters_block + model_block + generated_quantities_block

    return stan_model



## ============================================================================================================================================================
## ============================================================================================================================================================
## ============================================================================================================================================================



def generate_stan_code_population_shared_parameters(within_session_timeconstants : list,
                                  session_interaction_types : list,
                                  context_depth : int,
                                  same_nback_depth : int,
                                  repeat_bias_in_connection_weights : bool = False) -> str:
    context_depth = max(0, int(context_depth))
    same_nback_depth = max(0, int(same_nback_depth))

    within_session_timeconstants = [str(ss).replace(' ', '_') for ss in within_session_timeconstants]
    session_interaction_types = [str(ss).replace(' ', '_') for ss in session_interaction_types]

    assert same_nback_depth <= 1, "can't generate model with nback biases greater than 1"
    lb = "{"
    rb = "}"

    data_block = """
data {
    int N; // Number of data points
    int M; // number of possible observations
    array[N] int Y;
    array[N] int local_timeconstant_id;
    array[N] int session_id;
    array[N] real local_time;

    int T; // max observation distance
    int K; // number of subjects that are stacked on top of each other
    array[K+1] int subject_start_idx; // first element should be 1, last element should be N+1

    real prior_alpha_shape;
    real prior_alpha_scale;

    int P_alpha;
    matrix[N,P_alpha] alpha_loadings;
"""
    

    
    for ii in session_interaction_types:
         data_block += f"""    array[N,2] real session_time_{ii}; // column 1: projecting time, column 2: receiving time; if negative, doesn't do either
"""
    
    if(len(within_session_timeconstants) > 0):
        data_block += """
    real prior_timeconstant_within_session_shape;
    real prior_timeconstant_within_session_scale;
    int P_within_session_timeconstants;
"""
        for ii in within_session_timeconstants:
            data_block += f"""    matrix[N,P_within_session_timeconstants] timeconstant_within_session_{ii}_loadings;
"""
    if(len(session_interaction_types) > 0):
        data_block += """    real prior_timeconstant_between_sessions_shape;
    real prior_timeconstant_between_sessions_scale;
    int P_between_session_timeconstants;
"""
        for ii in session_interaction_types:
            data_block += f"""    matrix[N,P_between_session_timeconstants] timeconstant_between_sessions_{ii}_loadings;
"""

    for ii in range(1,context_depth+1):
        data_block += f"""
    matrix[N,T] is_same_context_{ii};
    real prior_context_similarity_depth_{ii}_alpha;
    real prior_context_similarity_depth_{ii}_beta;
    int P_context_similarity_depth_{ii};
    matrix[N,P_context_similarity_depth_{ii}] context_similarity_depth_{ii}_loadings;
"""

    for ii in range(1,same_nback_depth+1):
        data_block += f"""
    matrix[N,T] is_same_{ii}_back;
    real prior_repeat_bias_{ii}_back_shape;
    int P_repeat_bias_{ii}_back;
    matrix[N,P_repeat_bias_{ii}_back] repeat_bias_{ii}_back_loadings;
"""
        
    data_block += """}
"""

    transformed_data_block = """
transformed data {
    // variables to turn main computation in matrix operations
    vector[T] vs = rep_vector(1, T);

    matrix[N,T] is_same_observation = rep_matrix(0, N, T); // for numerator in CRP likelihood p(y_t | y_1:t-1)
    matrix[N,T] is_prev_observation = rep_matrix(0, N, T); // for denominator in CRP likelihood p(y_t | y_1:t-1)
    for (kk in 1:K) {
        int start_t = subject_start_idx[kk];
        int end_t   = subject_start_idx[kk+1];
        int t_c     = end_t - start_t;
        for (aa in 1:t_c) {
            int aa_c = aa + start_t - 1;
            for (bb in 1:t_c) {
                int bb_c = bb + start_t - 1;
                if(local_time[bb_c] < local_time[aa_c]) {
                    is_prev_observation[aa_c,bb] = 1;
                }

                if((is_prev_observation[aa_c,bb] > 0) && (Y[aa_c] == Y[bb_c])) {
                    is_same_observation[aa_c,bb] = 1;
                }
            }
        }
    }
    
"""
    for ii in within_session_timeconstants:
        transformed_data_block += f"""    matrix[N,T] deltas_{ii} = rep_matrix(0, N, T);
"""
        
    for ii in session_interaction_types:
        transformed_data_block += f"""    matrix[N,T] deltas_session_{ii} = rep_matrix(0, N, T);
"""
        
    transformed_data_block += """
    for (kk in 1:K) {
        int start_t = subject_start_idx[kk];
        int end_t   = subject_start_idx[kk+1];
        int t_c     = end_t - start_t;
        for (aa in 1:t_c) {
            int aa_c = aa + start_t - 1;
            for (bb in 1:t_c) {
                int bb_c = bb + start_t - 1;
"""

    for id_num_0, ii in enumerate(within_session_timeconstants):
        id_num = id_num_0 + 1;
        transformed_data_block += f"""
                if((is_prev_observation[aa_c,bb] > 0) && (local_timeconstant_id[aa_c] == {id_num}) && (local_timeconstant_id[bb_c] == {id_num}) && (session_id[aa_c] == session_id[bb_c])) {lb}
                    deltas_{ii}[aa_c,bb] = local_time[aa_c]-local_time[bb_c];
                {rb}
"""

    for ii in session_interaction_types:
        transformed_data_block += f"""
                if((is_prev_observation[aa_c,bb] > 0)&& (session_time_{ii}[aa_c,2] > 0) && (session_time_{ii}[bb_c,1] > 0)) {lb}
                    deltas_session_{ii}[aa_c,bb] = session_time_{ii}[aa_c,2]-session_time_{ii}[bb_c,1];
                {rb}
"""
    transformed_data_block += """
            }
        }
    }
"""
    for ii in range(1,context_depth+1):
        transformed_data_block += f"""
    matrix[N,T] is_different_context_{ii} = rep_matrix(0, N, T);
"""

    if(context_depth > 0):
        transformed_data_block += """
    for (kk in 1:K) {
        int start_t = subject_start_idx[kk];
        int end_t   = subject_start_idx[kk+1];
        int t_c     = end_t - start_t;
        for (aa in 1:t_c) {
            int aa_c = aa + start_t - 1;
            for (bb in 1:t_c) {
                int bb_c = bb + start_t - 1;
"""

        for ii in range(1,context_depth+1):
            transformed_data_block +=f"""
                if((is_prev_observation[aa_c,bb] > 0) && (is_same_context_{ii}[aa_c,bb] <= 0)) {lb}
                    is_different_context_{ii}[aa_c,bb] = 1;
                {rb}
"""

        transformed_data_block += """
            }
        }
    }
"""

    if(same_nback_depth > 0):
        for ii in range(1,same_nback_depth+1):
            transformed_data_block += f"""
    vector[N] Y_is_same_as_{ii}_back = rep_vector(0, N);
    for (kk in 1:K) {lb}
        int start_t = subject_start_idx[kk];
        int end_t   = subject_start_idx[kk+1];
        int t_c     = end_t - start_t;
        for (aa in 1:t_c) {lb}
            int aa_c = aa + start_t - 1;
            if(is_same_{ii}_back[aa_c,aa] > 0) {lb}
                Y_is_same_as_{ii}_back[aa_c] = 1;
            {rb}
        {rb}
    {rb}
"""

    transformed_data_block += """
    // prior parameter transformations for computations
    real prior_alpha_scale_inv = inv(prior_alpha_scale);
"""

    if(len(within_session_timeconstants) > 0):
        transformed_data_block += """
    real prior_timeconstant_within_session_scale_inv   = inv(prior_timeconstant_within_session_scale);
"""
    if(len(session_interaction_types) > 0):
        transformed_data_block += """
    real prior_timeconstant_between_sessions_scale_inv = inv(prior_timeconstant_between_sessions_scale);
"""

    for ii in range(1,same_nback_depth+1):
        transformed_data_block += f"""
    real prior_repeat_bias_{ii}_back_scale     = inv(prior_repeat_bias_1_back_shape);
    real prior_repeat_bias_{ii}_back_scale_inv = prior_repeat_bias_1_back_shape;
"""
    transformed_data_block += """}
"""

    parameters_block = """parameters {
    vector<lower=0>[P_alpha] alpha;   
"""
    for ii in within_session_timeconstants: 
        parameters_block += f"""    vector<lower=0>[P_within_session_timeconstants] timeconstant_within_session_{ii};
"""

    for ii in session_interaction_types: 
        parameters_block += f"""    vector<lower=0>[P_between_session_timeconstants] timeconstant_between_sessions_{ii};
"""

    for ii in range(1,same_nback_depth+1): 
        parameters_block += f"""    vector<lower=0>[P_repeat_bias_{ii}_back] repeat_bias_{ii}_back;
"""

    for ii in range(1,context_depth+1): 
        parameters_block += f"""    vector<lower=0,upper=1>[P_context_similarity_depth_{ii}] context_similarity_depth_{ii};
"""

    parameters_block += """}
"""

    
    model_block = """model {
"""
    prob_seating = ""
    prob_seating += """
    matrix[N,T] weights_same_obs;
    matrix[N,T] weights_all_obs;
    vector[N] ps;

"""
    if(same_nback_depth > 0):
        depth = 1;
        prob_seating += f"""
    vector[N] BaseMeasure;
    vector[N] repeat_bias_1_back_0;
    repeat_bias_1_back_0 = repeat_bias_1_back_loadings * repeat_bias_1_back;
    BaseMeasure = (Y_is_same_as_{depth}_back .* (repeat_bias_1_back_0-1.0) + 1.0) ./ (repeat_bias_1_back_0 + (M-1.0));
"""
    else:
        prob_seating += """    real BaseMeasure;
    BaseMeasure = inv(M);
"""
    
    model_block += """
    alpha                         ~ gamma(prior_alpha_shape,                         prior_alpha_scale_inv);
"""
    prob_seating += f"""
    vector[N] alpha_0;
    alpha_0 = alpha_loadings * alpha;
"""
    
    for ii in within_session_timeconstants:
        model_block += f"""
    timeconstant_within_session_{ii}   ~ gamma(prior_timeconstant_within_session_shape,   prior_timeconstant_within_session_scale_inv);
"""
        prob_seating += f"""
    matrix[N,T] timeconstant_within_session_{ii}_0;
    timeconstant_within_session_{ii}_0 = rep_matrix(timeconstant_within_session_{ii}_loadings * timeconstant_within_session_{ii}, T);
"""

    for ii in session_interaction_types:
        model_block += f"""    
    timeconstant_between_sessions_{ii} ~ gamma(prior_timeconstant_between_sessions_shape, prior_between_sessions_scale_inv);
"""
        prob_seating += f"""
    matrix[N,T] timeconstant_between_sessions_{ii}_0;
    timeconstant_between_sessions_{ii}_0 = rep_matrix(timeconstant_between_sessions_{ii}_loadings * timeconstant_between_sessions_{ii}, T);
"""


    for ii in range(1, same_nback_depth+1):
        model_block += f"""
    repeat_bias_{ii}_back            ~ gamma(prior_repeat_bias_{ii}_back_shape,            prior_repeat_bias_{ii}_back_scale_inv);
"""

    for ii in range(1, context_depth+1):
        model_block += f"""
    context_similarity_depth_{ii}  ~ beta(prior_context_similarity_depth_{ii}_alpha,   prior_context_similarity_depth_{ii}_beta);
    """
        prob_seating += f"""
    matrix[N,T] context_similarity_depth_{ii}_0;
    context_similarity_depth_{ii}_0 = rep_matrix(context_similarity_depth_{ii}_loadings * context_similarity_depth_{ii}, T);
"""
        
    prob_seating +=     f"""
    weights_all_obs   = is_prev_observation .* exp("""
    
    space_str = ""
    space_str_b = "                                                   "
    for ii in within_session_timeconstants:
        prob_seating += f"""{space_str} -(deltas_{ii}./timeconstant_within_session_{ii}_0)
"""
        space_str = space_str_b

    for ii in session_interaction_types:
        prob_seating += f"""{space_str} -(deltas_session_{ii}./timeconstant_within_session_{ii}_0)
"""
        space_str = space_str_b
        
    for ii in range(1, context_depth+1):
        prob_seating += f"""{space_str}+ (log1m(context_similarity_depth_{ii}_0) .* is_different_context_{ii})
"""
        space_str = space_str_b

    if(repeat_bias_in_connection_weights):    
        for ii in range(1, same_nback_depth+1):
            prob_seating += f"""{space_str}+ (log(context_repeat_bias_{ii}_back_0)  .* is_same_{ii}_back)
    """
            space_str = space_str_b

            
    prob_seating += f"""{space_str});
"""
        
    prob_seating += """
    weights_same_obs  = is_same_observation .* weights_all_obs;

    // probability of drawing the observed observations given their pasts
    ps =  ((weights_same_obs * vs) + (alpha_0.*BaseMeasure)) ./  ((weights_all_obs * vs) + (alpha_0));
    """

    model_block += prob_seating;
    model_block += """
    1 ~ bernoulli(ps); // note: not generative - this is a simplified distribution to make the log likelihood computations work quickly in Stan
}
"""

    generated_quantities_block = f"""
generated quantities {lb}
    vector[K] log_likelihood;
    {lb}
        {prob_seating}
        for(kk in 1:K) {lb}
            int start_t = subject_start_idx[kk];
            int end_t   = subject_start_idx[kk+1]-1;
            log_likelihood[kk] = bernoulli_lpmf(1 | ps[start_t:end_t]);
        {rb}
    {rb}
{rb}
"""
    transformed_parameters_block = ""




    stan_model = data_block + transformed_data_block + parameters_block + transformed_parameters_block + model_block + generated_quantities_block

    return stan_model

