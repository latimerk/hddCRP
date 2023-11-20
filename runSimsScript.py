# %%
import nest_asyncio
nest_asyncio.apply()

from hddCRP.simulations import simulate_sessions
from hddCRP.modelBuilder import cdCRP
import numpy as np
import pandas as pd

import arviz as az

import os
import sys

# %%

args = [-1, 0, 1, 10]
if __name__ == "__main__":
    for ii, arg in enumerate(sys.argv):
        if(ii < len(args)):
            args[ii] = arg
simulation_range = [int(args[1])];

min_blocks = int(args[2]);
max_blocks = int(args[3]);

block_range = range(min_blocks, max_blocks+1)
run_range = range(0, 50)

results_directory = "Results/Simulations/"
OVERWRITE = False;


if(not os.path.exists(results_directory)):
    os.makedirs(results_directory)

nback_depth   = 1;
context_depth = 1;
include_tau = True;



# %%


for simulation_id in simulation_range:

    if(simulation_id == 0):
        alpha = 2
        different_context_weights = [0.7, 0.7];
        within_session_timescales  = {"A" : 20}
        between_session_timescales = None; #{("A","A") : 2}
        repeat_bias_1_back = 0.75

        session_labels  = lambda n_blocks : ["A"] ;
        session_lengths = lambda n_blocks : [10 * n_blocks];
        num_subjects    = lambda n_blocks : 1;
        num_responses = 3
    elif(simulation_id == 1):
        alpha = 5
        different_context_weights = [0.3, 0.3];
        within_session_timescales  = {"A" : 40}
        between_session_timescales =  None; #{("A","A") : 2}
        repeat_bias_1_back = 1.0

        session_labels  = lambda n_blocks : ["A"] ;
        session_lengths = lambda n_blocks : [10 * n_blocks];
        num_subjects = lambda n_blocks : 1;
        num_responses = 3
    elif(simulation_id == 2):
        alpha = 5
        different_context_weights = [0.2, 0.2];
        within_session_timescales  = {"A" : 20}
        between_session_timescales = None; #{("A","A") : 2}
        repeat_bias_1_back = 0.8

        session_labels  = lambda n_blocks : ["A"] ;
        session_lengths = lambda n_blocks : [10 * n_blocks];
        num_subjects    = lambda n_blocks : 1;
        num_responses = 3
    elif(simulation_id == 3):
        alpha = 5
        different_context_weights = [0.8, 0.8];
        within_session_timescales  = {"A" : 40}
        between_session_timescales =  None; #{("A","A") : 2}
        repeat_bias_1_back = 1.0

        session_labels  = lambda n_blocks : ["A"] ;
        session_lengths = lambda n_blocks : [10 * n_blocks];
        num_subjects = lambda n_blocks : 1;
        num_responses = 3
    elif(simulation_id == 4):
        alpha = 2
        different_context_weights = [0.7, 0.7];
        within_session_timescales  = {"A" : 20}
        between_session_timescales = None; #{("A","A") : 2}
        repeat_bias_1_back = 0.75

        session_labels  = lambda n_blocks : ["A"] ;
        session_lengths = lambda n_blocks : [10 * n_blocks ];
        num_subjects = lambda n_blocks : 10;
        num_responses = 3
    elif(simulation_id == 5):
        alpha = 5
        different_context_weights = [0.3, 0.3];
        within_session_timescales  = {"A" : 40}
        between_session_timescales = None; #{("A","A") : 2}
        repeat_bias_1_back = 1.0

        session_labels  = lambda n_blocks : ["A"] ;
        session_lengths = lambda n_blocks : [10 * n_blocks ];
        num_subjects = lambda n_blocks :  10;
        num_responses = 3
    elif(simulation_id == 6):
        alpha = 5
        different_context_weights = [0.2, 0.2];
        within_session_timescales  = {"A" : 20}
        between_session_timescales = None; #{("A","A") : 2}
        repeat_bias_1_back = 0.8

        session_labels  = lambda n_blocks : ["A"] ;
        session_lengths = lambda n_blocks : [10 * n_blocks ];
        num_subjects = lambda n_blocks : 10;
        num_responses = 3
    elif(simulation_id == 7):
        alpha = 5
        different_context_weights = [0.8, 0.8];
        within_session_timescales  = {"A" : 40}
        between_session_timescales = None; #{("A","A") : 2}
        repeat_bias_1_back = 1.0

        session_labels  = lambda n_blocks : ["A"] ;
        session_lengths = lambda n_blocks : [10 * n_blocks ];
        num_subjects = lambda n_blocks :  10;
        num_responses = 3
    elif(simulation_id == 8):
        alpha = 5
        different_context_weights = [0.2, 0.2];
        within_session_timescales  = {"A" : 20}
        between_session_timescales = None; #{("A","A") : 2}
        repeat_bias_1_back = 0.5

        session_labels  = lambda n_blocks : ["A"] ;
        session_lengths = lambda n_blocks : [25 * n_blocks ];
        num_subjects = lambda n_blocks : 1;
        num_responses = 3
    elif(simulation_id == 9):
        alpha = 3
        different_context_weights = [0.8, 0.8];
        within_session_timescales  = {"A" : 50}
        between_session_timescales = None; #{("A","A") : 2}
        repeat_bias_1_back = 1.0

        session_labels  = lambda n_blocks : ["A"] ;
        session_lengths = lambda n_blocks : [25 * n_blocks ];
        num_subjects = lambda n_blocks :  1;
        num_responses = 3
    elif(simulation_id == 10):
        alpha = 5
        different_context_weights = [0.2, 0.2];
        within_session_timescales  = {"A" : 20}
        between_session_timescales = None; #{("A","A") : 2}
        repeat_bias_1_back = 0.5

        session_labels  = lambda n_blocks : ["A"] ;
        session_lengths = lambda n_blocks : [25 * n_blocks ];
        num_subjects = lambda n_blocks : 10;
        num_responses = 3
    elif(simulation_id == 11):
        alpha = 3
        different_context_weights = [0.8, 0.8];
        within_session_timescales  = {"A" : 50}
        between_session_timescales = None; #{("A","A") : 2}
        repeat_bias_1_back = 1.0

        session_labels  = lambda n_blocks : ["A"] ;
        session_lengths = lambda n_blocks : [25 * n_blocks ];
        num_subjects = lambda n_blocks :  10;
        num_responses = 3
    elif(simulation_id == 12):
        alpha = 5
        different_context_weights = [0.2, 0.2];
        within_session_timescales  = {"A" : 20}
        between_session_timescales = None; #{("A","A") : 2}
        repeat_bias_1_back = 0.5

        session_labels  = lambda n_blocks : ["A"] ;
        session_lengths = lambda n_blocks : [25 * n_blocks ];
        num_subjects = lambda n_blocks : 1;
        num_responses = 8
    elif(simulation_id == 13):
        alpha = 3
        different_context_weights = [0.8, 0.8];
        within_session_timescales  = {"A" : 50}
        between_session_timescales = None; #{("A","A") : 2}
        repeat_bias_1_back = 1.0

        session_labels  = lambda n_blocks : ["A"] ;
        session_lengths = lambda n_blocks : [25 * n_blocks ];
        num_subjects = lambda n_blocks :  1;
        num_responses = 8
    else:
        raise NotImplementedError("No sim found")
    
    if(nback_depth < 1):
        repeat_bias_1_back = None;
    different_context_weights = different_context_weights[:context_depth]
    if(not include_tau):
        within_session_timescales  = {"A" : np.inf}

    for block_idx in block_range:
        print(f"BLOCK {block_idx}")
        fit_file = f"{results_directory}/sim_{simulation_id}_block_{block_idx}"
        fit_summary_file = f"{results_directory}/sim_summary_{simulation_id}_block_{block_idx}"
        fit_file += f"_cd{context_depth}_nb{nback_depth}"
        fit_summary_file  += f"_cd{context_depth}_nb{nback_depth}"
        if(not include_tau):
            fit_file += f"_no_tau"
            fit_summary_file  += f"_no_tau"
        fit_file += f".pkl"
        fit_summary_file  += f".pkl"


        if(((not os.path.isfile(fit_file)) or (not os.path.isfile(fit_summary_file))) or OVERWRITE):
            simulation_fits = pd.DataFrame()
            simulation_fit_metrics = pd.DataFrame()
            for run_idx in run_range:
                sim_seed  = (simulation_id+1) * 10000 + nback_depth * 1001 + context_depth * 1000 + run_idx*100 
                stan_seed = (simulation_id+1) * 10000 + nback_depth * 1001 + context_depth * 1000 + run_idx*100 + block_idx
                # sim_rng = np.random.Generator(np.random.MT19937(sim_seed))
                
                seqs = [];
                subject_labels = [];
                session_labels_all = [];
                for jj in range(num_subjects(block_idx)):
                    sim_rng = np.random.Generator(np.random.MT19937(sim_seed + jj))
                    seqs_c = simulate_sessions(session_lengths=session_lengths(block_idx), session_labels=session_labels(block_idx), num_responses=num_responses, 
                                            alpha=alpha,
                                            different_context_weights=different_context_weights,
                                            within_session_timescales=within_session_timescales, between_session_timescales=between_session_timescales,
                                            repeat_bias_1_back=repeat_bias_1_back, rng=sim_rng)
                    subject_labels += [jj] * len(seqs_c)
                    session_labels_all += session_labels(block_idx)
                    seqs += seqs_c;
                print(seqs)
                print(f"BLOCK {block_idx} - RUN {run_idx}")

                model = cdCRP(seqs, subject_labels=subject_labels, session_labels=session_labels_all, possible_observations=list(range(num_responses)));
                model.same_nback_depth = nback_depth;
                model.context_depth = context_depth;
                model.within_session_decay_enabled = include_tau
                
                model.context_depth = len(different_context_weights)
                model.build(random_seed=stan_seed);
                model.fit_model()

                fit_df  = model.fit.to_frame()
                # map_fit = model.get_map()
                fit_df["block"] = block_idx
                fit_df["run"]   = run_idx
                fit_df["simulation"]   = simulation_id
                summary_df = model.fit_summary()
                summary_df["block"] = block_idx
                summary_df["run"]   = run_idx
                # summary_df["MAP"] = pd.Series(map_fit)
                summary_df["simulation"]   = simulation_id

                true_param = {"alpha" : alpha}
                if(include_tau):
                    true_param["timeconstant_within_session_A"] = within_session_timescales["A"]
                if(nback_depth >= 1):
                    true_param["repeat_bias_1_back"] = repeat_bias_1_back
                if(context_depth >= 1):
                    true_param["context_similarity_depth_1"] = different_context_weights[0]
                if(context_depth >= 2):
                    true_param["context_similarity_depth_2"] = different_context_weights[1]

                summary_df["true"] = pd.Series(true_param)
                

                simulation_fit_metrics = pd.concat([simulation_fit_metrics,summary_df], copy=False)
                simulation_fits = pd.concat([simulation_fits,fit_df], copy=False)

            simulation_fits.to_pickle(fit_file)
            simulation_fit_metrics.to_pickle(fit_summary_file)
        else:
            print("Fit files found: not overriding")

