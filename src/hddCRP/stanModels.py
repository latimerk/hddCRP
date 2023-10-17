

def generate_stan_code_individual(within_session_timeconstants : list, session_interaction_types : list, context_depth : int, same_nback_depth : int) -> str:
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

    real prior_alpha_shape;
    real prior_alpha_scale;
"""
    
    for ii in session_interaction_types:
         data_block += f"""    array[N,2] real session_time_{ii}; // column 1: projecting time, column 2: receiving time; if negative, doesn't do either
"""
    
    if(len(within_session_timeconstants) > 0):
        data_block += """    real prior_timeconstant_within_session_shape;
    real prior_timeconstant_within_session_scale;
"""
    if(len(session_interaction_types) > 0):
        data_block += """    real prior_timeconstant_between_sessions_shape;
    real prior_timeconstant_between_sessions_scale;
"""

    for ii in range(1,context_depth+1):
        data_block += f"""    matrix[N,N] is_same_context_{ii};
"""
        data_block += f"""    real prior_context_similarity_depth_{ii}_alpha;
"""
        data_block += f"""    real prior_context_similarity_depth_{ii}_beta;
"""

    for ii in range(1,same_nback_depth+1):
        data_block += f"""    matrix[N,N] is_same_{ii}_back;
"""
        data_block += f"""    real prior_repeat_bias_{ii}_back_shape;
"""
    data_block += """}
"""

    transformed_data_block = """
transformed data {
    // variables to turn main computation in matrix operations
    vector[N] vs = rep_vector(1, N);

    matrix[N,N] is_same_observation = rep_matrix(0, N, N); // for numerator in CRP likelihood p(y_t | y_1:t-1)
    matrix[N,N] is_prev_observation = rep_matrix(0, N, N); // for denominator in CRP likelihood p(y_t | y_1:t-1)
    for (aa in 1:N) {
        for (bb in 1:N) {
            if(local_time[bb] < local_time[aa]) {
                is_prev_observation[aa,bb] = 1;
            }

            if((is_prev_observation[aa,bb] > 0) && (Y[aa] == Y[bb])) {
                is_same_observation[aa,bb] = 1;
            }
        }
    }
    
"""
    for ii in within_session_timeconstants:
        transformed_data_block += f"""    matrix[N,N] deltas_{ii} = rep_matrix(0, N, N);
"""
        
    for ii in session_interaction_types:
        transformed_data_block += f"""    matrix[N,N] deltas_session_{ii} = rep_matrix(0, N, N);
"""
        
    transformed_data_block += """    for (aa in 1:N) {
        for (bb in 1:N) {
"""

    for id_num_0, ii in enumerate(within_session_timeconstants):
        id_num = id_num_0 + 1;
        transformed_data_block += f"""
            if((bb < aa) && (local_timeconstant_id[aa] == {id_num}) && (local_timeconstant_id[bb] == {id_num}) && (session_id[aa] == session_id[bb])) {lb}
                deltas_{ii}[aa,bb] = local_time[aa]-local_time[bb];
            {rb}
"""

    for ii in session_interaction_types:
        transformed_data_block += f"""
            if((bb < aa) && (session_time_{ii}[aa,2] > 0) && (session_time_{ii}[bb,1] > 0)) {lb}
                deltas_session_{ii}[aa,bb] = session_time_{ii}[aa,2]-session_time_{ii}[bb,1];
            {rb}
"""
    transformed_data_block += """        }
    }
"""
    for ii in range(1,context_depth+1):
        transformed_data_block += f"""    matrix[N,N] is_different_context_{ii} = rep_matrix(0, N, N);
"""

    if(context_depth > 0):
        transformed_data_block += """
    for (aa in 1:N) {
        for (bb in 1:N) {
"""

        for ii in range(1,context_depth+1):
            transformed_data_block +=f"""
            if(bb < aa && is_same_context_{ii}[aa,bb] <= 0) {lb}
                is_different_context_{ii}[aa,bb] = 1;
            {rb}
"""

        transformed_data_block += """
        }
    }
"""

    if(same_nback_depth > 0):
        for ii in range(1,same_nback_depth+1):
            transformed_data_block += f"""    vector[N] Y_is_same_as_{ii}_back = rep_vector(0, N);
    for (aa in 1:N) {lb}
        if(is_same_{ii}_back[aa,aa] > 0) {lb}
            Y_is_same_as_{ii}_back[aa] = 1;
        {rb}
    {rb}
"""

    transformed_data_block += """
    // prior parameter transformations for computations
    real prior_alpha_scale_log = log(prior_alpha_scale);
    real prior_alpha_scale_inv = inv(prior_alpha_scale);
"""

    if(len(within_session_timeconstants) > 0):
        transformed_data_block += """    real prior_timeconstant_within_session_scale_log   = log(prior_timeconstant_within_session_scale);
    real prior_timeconstant_within_session_scale_inv   = inv(prior_timeconstant_within_session_scale);
"""
    if(len(session_interaction_types) > 0):
        transformed_data_block += """    real prior_timeconstant_between_sessions_scale_log = log(prior_timeconstant_between_sessions_scale);
    real prior_timeconstant_between_sessions_scale_inv = inv(prior_timeconstant_between_sessions_scale);
"""

    for ii in range(1,same_nback_depth+1):
        transformed_data_block += f"""
    real prior_repeat_bias_{ii}_back_scale     = inv(prior_repeat_bias_1_back_shape);
    real prior_repeat_bias_{ii}_back_scale_inv = prior_repeat_bias_1_back_shape;
    real prior_repeat_bias_{ii}_back_scale_log = log(prior_repeat_bias_1_back_scale);
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

    
    model_block = """model {
    matrix[N,N] weights_same_obs;
    matrix[N,N] weights_all_obs;
    vector[N] ps;
"""
    if(same_nback_depth > 0):
        depth = 1;
        model_block += f"""    vector[N] BaseMeasure;
    BaseMeasure = ((repeat_bias_1_back-1.0) * Y_is_same_as_{depth}_back + 1.0) / (repeat_bias_1_back + (M-1.0));
"""
    else:
        model_block += """    real BaseMeasure;
    BaseMeasure = inv(M);
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
        
    model_block +=     f"""
    weights_all_obs   = is_prev_observation .* exp("""
    
    space_str = ""
    space_str_b = "                                                   "
    for ii in within_session_timeconstants:
        model_block += f"""{space_str} (- deltas_{ii}/timeconstant_within_session_{ii})
"""
        space_str = space_str_b

    for ii in session_interaction_types:
        model_block += f"""{space_str} (- deltas_session_{ii}/timeconstant_within_session_{ii})
"""
        space_str = space_str_b
        
    for ii in range(1, context_depth+1):
        model_block += f"""{space_str}+ (log(context_similarity_depth_{ii})  * is_different_context_{ii})
"""
        space_str = space_str_b
        
    for ii in range(1, same_nback_depth+1):
        model_block += f"""{space_str}+ (log(context_repeat_bias_{ii}_back)   * is_same_{ii}_back)
 """
        space_str = space_str_b
    model_block += f"""{space_str});
"""
        
    model_block += """
    weights_same_obs  = is_same_observation .* weights_all_obs;

    // probability of drawing the observed observations given their pasts
    ps =  ((weights_same_obs * vs) + (alpha*BaseMeasure)) ./  ((weights_all_obs * vs) + (alpha));
    """

    
    # if(same_nback_depth > 0):
    #     bm_idx = "[ii]"
    # else:
    #     bm_idx = ""

    # model_block += f"""
    # for (ii in 1:N) {lb}
    #     ps[ii] =(sum(weights_same_obs[:ii-1]) + alpha*BaseMeasure{bm_idx}) / (sum(weights_all_obs[:ii-1]) + alpha);
    # {rb}
    # """
        
    model_block += """
    1 ~ bernoulli(ps); // note: not generative - this is a simplified distribution to make the log likelihood computations work quickly in Stan
}
"""
    transformed_parameters_block = ""
    stan_model = data_block + transformed_data_block + parameters_block + transformed_parameters_block + model_block

    return stan_model



## ============================================================================================================================================================
## ============================================================================================================================================================
## ============================================================================================================================================================
model_individual_session_context_2_repeat_1 = """
"""