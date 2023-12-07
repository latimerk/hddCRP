# %%
import nest_asyncio
nest_asyncio.apply()

from hddCRP.simulations import sim_markov, stack_distributions, interp_distributions, construct_default_hmm
from hddCRP.modelBuilder import cdCRP
import numpy as np
import pandas as pd

import arviz as az

import os
import sys
import time

# %%
def generate_ps(sim, block, alpha_pre = 0, alpha_post = 0):
    T_total = 25 * block;

    T_pre  = int(T_total/2)
    T_post = T_total - T_pre

    p_pre, t_pre, q_pre, p_post, t_post, q_post = construct_default_hmm(alpha_pre=alpha_pre, alpha_post = alpha_post)
    
    
    sim = sim - 20

    p_uniform = np.ones_like(p_pre)
    p_uniform /= p_uniform.sum()

    match sim:
        case 0:
            return stack_distributions([T_pre, T_post], [p_pre, p_post]), p_pre
        case 1:
            return stack_distributions([T_pre, T_post], [t_pre, t_post]), p_pre
        case 2:
            return interp_distributions([T_total], [p_pre, p_post]), p_pre
        case 3:
            return interp_distributions([T_total], [t_pre, t_post]), p_pre
        case 4:
            return stack_distributions([T_total], [p_pre]), p_pre
        case 5:
            return stack_distributions([T_total], [t_pre]), p_pre
        case 6:
            return stack_distributions([T_total], [p_post]), p_post
        case 7:
            return stack_distributions([T_total], [t_post]), p_post
        case 8:
            return stack_distributions([T_pre, T_post], [q_pre, q_post]), p_pre
        case 9:
            return interp_distributions([T_total], [q_pre, q_post]), p_pre
        case 10:
            return stack_distributions([T_total], [q_pre]), p_pre
        case 11:
            return stack_distributions([T_total], [q_post]), p_post
        
        case 12:
            return stack_distributions([T_pre, T_post], [p_pre, t_post]), p_pre
        case 13:
            return interp_distributions([T_total], [p_pre, t_post]), p_pre
        case 14:
            return stack_distributions([T_pre, T_post], [p_pre, q_post]), p_pre
        case 15:
            return interp_distributions([T_total], [p_pre, q_post]), p_pre
        case 16:
            return stack_distributions([T_pre, T_post], [t_pre, q_post]), p_pre
        case 17:
            return interp_distributions([T_total], [t_pre, q_post]), p_pre
        case 18:
            return stack_distributions([T_pre, T_post], [p_uniform, q_post]), p_uniform

# %%
args = [-1, 20, 37]


narg = 0;
inputs = [];
if __name__ == "__main__":
    for ii, arg in enumerate(sys.argv):
        if(ii > 0):
            # args[ii] = int(arg)
            inputs += [int(arg)]
            narg += 1;
interp  = [33, 35, 37, 29, 22, 23]
stacked = [32, 34, 36, 28, 20, 21, 38]
if(narg >= 1):
    simulation_range = inputs;
else:
    simulation_range = interp;
print(f"simulation_range = {simulation_range}")
time.sleep(2.0)

block_range = range(1,8+1)
block_range = [4]
run_range = range(50)
# dist_alpha = 0.1

results_directory = "Results/Simulations/"
OVERWRITE = False;


if(not os.path.exists(results_directory)):
    os.makedirs(results_directory)
num_responses = 3;

nback_depth   = 1;
context_depth = 1;
include_tau = True;

# %%
for simulation_id in simulation_range:
    for block_idx in block_range:
        if(block_idx == 2):
            dist_alpha_pre  = 0.1;
            dist_alpha_post = 0.1;
            num_blocks_for_sim = 2;
        elif(block_idx == 3):
            dist_alpha_pre = 0.5;
            dist_alpha_post = 0.1;
            num_blocks_for_sim = 2;
        elif(block_idx == 4):
            dist_alpha_pre  = 0.1;
            dist_alpha_post = 0.1;
            num_blocks_for_sim = 4;
        else:
            raise RuntimeError("invalid block")

        print(f"SIMULATION {simulation_id} - BLOCK {block_idx}")
        obs_probs, init_obs_probs = generate_ps(simulation_id, num_blocks_for_sim, dist_alpha_pre, dist_alpha_post)

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
                sim_rng = np.random.Generator(np.random.MT19937(sim_seed))
                
                print(f"init_obs_probs: {init_obs_probs}")
                print(f"obs_probs[0]: {obs_probs[0,...]}")
                print(f"obs_probs[end]: {obs_probs[-1,...]}")
                print(f"SIMULATION {simulation_id} - BLOCK {block_idx} - RUN {run_idx}")
                seqs = [sim_markov(obs_probs, init_obs_probs, rng=sim_rng)];
                subject_labels = [0];
                session_labels_all = ["A"];
                print(seqs)

                model = cdCRP(seqs, subject_labels=subject_labels, session_labels=session_labels_all, possible_observations=list(range(num_responses)));
                model.same_nback_depth = nback_depth;
                model.context_depth = context_depth;
                model.within_session_decay_enabled = include_tau
                
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

                

                simulation_fit_metrics = pd.concat([simulation_fit_metrics,summary_df], copy=False)
                simulation_fits = pd.concat([simulation_fits,fit_df], copy=False)

            simulation_fits.to_pickle(fit_file)
            simulation_fit_metrics.to_pickle(fit_summary_file)
        else:
            print("Fit files found: not overriding")



