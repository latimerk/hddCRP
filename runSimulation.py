# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import hddCRP.simulations
import hddCRP.modelFitting
import hddCRP.behaviorDataHandlers
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
import seaborn as sns
import sys

simulation_id = int(sys.argv[1]); 
run_num = int(sys.argv[2])
if(len(sys.argv) > 3):
    run_num_end = int(sys.argv[3])+1
else:
    run_num_end = run_num+1
run_range = range(run_num,run_num_end)

print("Running simulation " + str(simulation_id))
print("Run index " + str(run_num) + " to " + str(run_num_end-1))


# %%
results_directory = "Results/Simulations"
if(not os.path.exists(results_directory)):
    os.makedirs(results_directory)
    

overwrite_existing_results = False


num_warmup_samples = 5000
num_samples = 20000

initialize_fit_with_real_connections = False;

prior_scales = None
prior_shapes = None
use_nonsequential_filter_model = False
single_concentration_parameter=False
if(simulation_id == 0):
    depth  = 3; # look 2 actions in the past
    alphas = [5,10,10] # concentration parameters: per depth in the hddCRP tree. alphas[0] first level (no action context), alphas[1] is the second (for regularizing p(y_t | y_{t-1})), etc...
    between_session_time_constants = np.array([[ 1]]) # units = sessions
    within_session_time_constant = [20] # units = actions
    session_length = lambda run_idx, block_idx : 50 * (block_idx) # trials per session
    num_sessions   = lambda run_idx, block_idx : 1 # trials per session

    action_labels = [0,1,2] 
    maze_symbols = ['A']
    uniform_prior = False
    min_blocks_per_type = 1
    max_blocks_per_type = 10;
    prior_scales = {"alpha" : 5, "tau_within" : 25, "tau_between" : 5}
    prior_shapes = {"alpha" : 2, "tau_within" :  2, "tau_between" : 2}

    use_nonsequential_filter_model = False
    single_concentration_parameter = False;
    fit_depth = depth;
elif(simulation_id == 1):
    depth  = 3; # look 2 actions in the past
    alphas = [10,5,5] # concentration parameters: per depth in the hddCRP tree. alphas[0] first level (no action context), alphas[1] is the second (for regularizing p(y_t | y_{t-1})), etc...
    between_session_time_constants = np.array([[ 1]]) # units = sessions
    within_session_time_constant = [40] # units = actions
    session_length = lambda run_idx, block_idx : 50 * (block_idx) # trials per session
    num_sessions   = lambda run_idx, block_idx : 1 # trials per session

    action_labels = [0,1,2] 
    maze_symbols = ['A']
    uniform_prior = False
    min_blocks_per_type = 1
    max_blocks_per_type = 10;
    prior_scales = {"alpha" : 5, "tau_within" : 25, "tau_between" : 5}
    prior_shapes = {"alpha" : 2, "tau_within" :  2, "tau_between" : 2}

    use_nonsequential_filter_model = False
    single_concentration_parameter = False;
    fit_depth = depth;
elif(simulation_id == 2):
    depth  = 3; # look 2 actions in the past
    alphas = [10,5,np.inf] # concentration parameters: per depth in the hddCRP tree. alphas[0] first level (no action context), alphas[1] is the second (for regularizing p(y_t | y_{t-1})), etc...
    between_session_time_constants = np.array([[ 1]]) # units = sessions
    within_session_time_constant = [50] # units = actions
    session_length = lambda run_idx, block_idx : 50 * (block_idx) # trials per session
    num_sessions   = lambda run_idx, block_idx : 1 # trials per session

    action_labels = [0,1,2] 
    maze_symbols = ['A']
    uniform_prior = False
    min_blocks_per_type = 1
    max_blocks_per_type = 10;
    prior_scales = {"alpha" : 5, "tau_within" : 25, "tau_between" : 5}
    prior_shapes = {"alpha" : 2, "tau_within" :  2, "tau_between" : 2}

    use_nonsequential_filter_model = False
    single_concentration_parameter = False;

    fit_depth = depth;
elif(simulation_id == 3):
    depth  = 3; # look 2 actions in the past
    alphas = [10,5,5] # concentration parameters: per depth in the hddCRP tree. alphas[0] first level (no action context), alphas[1] is the second (for regularizing p(y_t | y_{t-1})), etc...
    between_session_time_constants = np.array([[ 1]]) # units = sessions
    within_session_time_constant = [50] # units = actions
    session_length = lambda run_idx, block_idx : 50 * (block_idx) # trials per session
    num_sessions   = lambda run_idx, block_idx : 1 # trials per session

    action_labels = [0,1,2] 
    maze_symbols = ['A']
    uniform_prior = False
    min_blocks_per_type = 1
    max_blocks_per_type = 10;
    prior_scales = {"alpha" : 5, "tau_within" : 25, "tau_between" : 5}
    prior_shapes = {"alpha" : 2, "tau_within" :  2, "tau_between" : 2}

    use_nonsequential_filter_model = False
    single_concentration_parameter = False;

    fit_depth = 2;
else:
    raise NotImplementedError

max_blocks_per_type = 5;
blocks_range = range(min_blocks_per_type,max_blocks_per_type+1);


true_parameters = {}
alpha_strs = ["no", "one_back", "two_back", "three_back"]
for dd in range(depth):
    true_parameters["alpha_concentration_" + alpha_strs[dd] + "_context"] = alphas[dd]
for aa_i, aa in enumerate(maze_symbols): 
    true_parameters["within_session_" + aa + "_time_constant"] = within_session_time_constant[aa_i]

if(max_blocks_per_type > 1):
    for aa_i, aa in enumerate(maze_symbols):
        if(num_sessions(run_range[0], blocks_range[0]) > 1):
            true_parameters[aa + "_to_" + aa + "_session_time_constant"] = between_session_time_constants[aa_i,aa_i]


        for bb_i, bb in enumerate(maze_symbols[aa_i+1:]):
            true_parameters[aa + "_to_" + bb + "_session_time_constant"] = between_session_time_constants[aa_i,bb_i]

# %%
for run_idx in run_range:
    print("run " + str(run_idx) + " in [" + str(run_range[0]) + ", " + str(run_range[-1]) + "]")
    for block_idx in blocks_range:
        print("block " + str(block_idx) + " in [" + str(blocks_range[0]) + ", " + str(blocks_range[-1]) + "]")
        filename = "{results_directory}/Sim_{sim_num}_block_{block_idx}_run_{run_idx}.pkl".format(results_directory=results_directory, sim_num=simulation_id, block_idx=block_idx, run_idx=run_idx)
        if(not os.path.isfile(filename) or overwrite_existing_results):
            rng_seed_sim = block_idx*0 + 100 + 10000*run_idx;
            rng_seed_fit = block_idx*0 + 200 + 10000*run_idx;
            rng_sim = np.random.Generator(np.random.MT19937(rng_seed_sim))
            rng_fit = np.random.Generator(np.random.MT19937(rng_seed_fit))


            session_lengths = [session_length(run_idx, block_idx)] * (len(maze_symbols) * num_sessions(run_idx, block_idx))
            session_labels = [];
            for aa in maze_symbols:
                session_labels += [aa] * num_sessions(run_idx, block_idx)  # which maze
            print("session_lengths: " + str(session_lengths))
            print("session_labels : " + str(session_labels))

            seqs, connection_data = hddCRP.simulations.simulate_sequential_hddCRP(session_lengths, session_labels, action_labels, depth, rng_sim, alphas, 
                    between_session_time_constants = between_session_time_constants, within_session_time_constant = within_session_time_constant)

            simulation_info = {"rng_seed_simulation" : rng_seed_sim, "rng_seed_fitting" : rng_seed_fit, "rng_type" : "MT19937",
                            "session_lengths" : session_lengths, "session_labels" : session_labels, "action_labels" : action_labels,
                            "seqs" : seqs, "connection_data" : connection_data}

            model = hddCRP.simulations.create_hddCRPModel_from_simulated_sequential_hddCRP(seqs, connection_data, rng=rng_fit, 
                    use_real_parameters=initialize_fit_with_real_connections, depth=fit_depth)

            

            
            tau_names = [str(xx) for xx in model.weight_param_labels]
            alphas_names = ["alpha_concentration_no_context", "alpha_concentration_one_back_context", "alpha_concentration_two_back_context"]
            model, samples, step_size_settings = hddCRP.behaviorDataHandlers.sample_model_for_maze_data(model, num_samples=num_samples, 
                            num_warmup_samples=num_warmup_samples, print_every=2500, uniform_prior=uniform_prior, prior_shapes=prior_shapes, prior_scales=prior_scales, 
                            single_concentration_parameter=single_concentration_parameter)

            if(depth > fit_depth):
                ag =  np.ones((depth-fit_depth, samples["alphas"].shape[1]));
                ag.fill(np.inf)
                samples["alphas"] = np.concatenate([samples["alphas"], ag], axis=0)

            MCMC_info = {"initialized_with_true_connections" : initialize_fit_with_real_connections,
                        "step_size_settings" : step_size_settings.to_dict(),
                        "num_warmup_samples" : num_warmup_samples,
                        "num_samples" : num_samples,
                        "uniform_prior" : uniform_prior,
                        "use_nonsequential_filter_model" : use_nonsequential_filter_model,
                        "prior_shapes" : prior_shapes,
                        "prior_scales" : prior_scales}
            samples["tau_parameter_names"] = tau_names
            samples["alphas_names"] = alphas_names[:depth]
            
            # save results to filename
            with open(filename, "wb") as results_file:
                results_data = {"true_parameters" : true_parameters,
                                "simulation_info" : simulation_info,
                                "MCMC_info" : MCMC_info,
                                "samples" : samples}
                pickle.dump(results_data, results_file)



# %%
