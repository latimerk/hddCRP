# %%
import nest_asyncio
nest_asyncio.apply()

from hddCRP.modelBuilder import cdCRP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import itertools
import time

import os

# %%
session_numbers = None;#[1]; # index by 1
number_of_trials = 50

overwrite_existing_results = False
results_directory = "Results/individualFit_fixed_alpha/"

if(not os.path.exists(results_directory)):
    os.makedirs(results_directory)

context_depth = 1;
nback_depth = 1;

tau_enabled = True

data_filename = 'data/Data_turns_all_by_session.pkl';
with open(data_filename, 'rb') as data_file:
    data = pickle.load(data_file)

subjects = list(data["data"].keys())
subjects.sort()
print("subjects = " + str(subjects))

alphas = np.arange(20,201,4)#np.arange(0.5, 20.1, 0.5)
action_labels = [0,1,2]

# %%
start_session = np.min(session_numbers)
end_session = np.max(session_numbers)
if(session_numbers is None):
    fit_file = f"{results_directory}/fits_trials_{number_of_trials}"
    fit_summary_file = f"{results_directory}/fit_summary_trials_{number_of_trials}"
    seed_offset = number_of_trials
else:
    start_session = np.min(session_numbers)
    end_session = np.max(session_numbers)
    fit_file = f"{results_directory}/fits_session_{start_session}"
    fit_summary_file = f"{results_directory}/fit_summary_session_{start_session}"
    if(end_session != start_session):
        fit_file += f"_to_{start_session}"
        fit_summary_file  += f"_to_{start_session}"
    seed_offset = start_session

fit_file += f"_cd{context_depth}_nb{nback_depth}"
fit_summary_file  += f"_cd{context_depth}_nb{nback_depth}"
if(not tau_enabled):
    fit_file += f"_no_tau"
    fit_summary_file  += f"_no_tau"

if(not os.path.isfile(fit_file) or overwrite_existing_results):
    data_fits = pd.DataFrame()
    data_fit_metrics = pd.DataFrame()
else:
    data_fits = pd.read_pickle(fit_file)
    data_fit_metrics = pd.read_pickle(fit_summary_file)
    print("fit file found")
for subject_index, subject in enumerate(subjects):
    print(f"subject {subject} ")
    sequences_0 = data["data"][subject]["data"]; # turns in each session
    session_types_0 = data["data"][subject]["task"] # which maze

    if(session_numbers is None):
        ii = list(np.where(np.array(session_types_0)=='C')[0])
        seqs_c = [sequences_0[xx] for xx in ii]
        seqs_c = list(itertools.chain.from_iterable(seqs_c))
        sequences = [seqs_c[:number_of_trials]]
        session_types = ['C']
    else:
        ii = list(np.where(np.array(session_types_0)=='C')[0][np.array(session_numbers)-1]) # sessions in map C
        sequences     = [sequences_0[xx] for xx in ii]
        session_types = [session_types_0[xx] for xx in ii]



    # build model with given sequences
    model = cdCRP(sequences, session_labels=session_types, subject_labels=subject, possible_observations=action_labels);

    # set model depth
    model.same_nback_depth = nback_depth
    model.context_depth = context_depth
    model.within_session_decay_enabled = tau_enabled

    need_to_save = False;

    for alpha_ii, alpha in enumerate(alphas):
        print(f"subject {subject_index}, alpha = {alpha}")

        if( ("subject" in data_fits) and ("alpha" in data_fits) and ("subject" in data_fit_metrics) and ("alpha" in data_fit_metrics) and
            data_fits.query(       "subject == @subject and alpha == @alpha").size > 0 and 
            data_fit_metrics.query("subject == @subject and alpha == @alpha").size > 0
            ):
            print("fit found.")
            continue;
        else:
            print("no fit found.")
            time.sleep(1)



        # fit with Stan
        stan_seed = (subject_index+1) * 1000 + alpha_ii*100 + seed_offset
        model.fixed_alpha = alpha
        model.build(random_seed=stan_seed);
        model.fit_model()

        # map_fit = model.get_map()
        fit_df  = model.fit.to_frame()
        summary_df = model.fit_summary()

        fit_df["subject"] = subject
        fit_df["alpha"] = alpha
        summary_df["subject"] = subject
        summary_df["alpha"] = alpha
        # summary_df["MAP"] = pd.Series(map_fit)

        if(session_numbers is None):
            summary_df["number_of_trials"] = number_of_trials
            summary_df["start_session_C"]  = pd.NA
            summary_df["end_session_C"]    = pd.NA
            fit_df["number_of_trials"] = number_of_trials
            fit_df["start_session_C"]  = pd.NA
            fit_df["end_session_C"]    = pd.NA
        else:
            summary_df["number_of_trials"] = pd.NA
            summary_df["start_session_C"]  = start_session
            summary_df["end_session_C"]    = end_session
            fit_df["number_of_trials"] = pd.NA
            fit_df["start_session_C"]  = start_session
            fit_df["end_session_C"]    = end_session

        data_fit_metrics = pd.concat([data_fit_metrics,summary_df], copy=False)
        data_fits = pd.concat([data_fits,fit_df], copy=False)
        need_to_save = True;

    if(need_to_save):
        print("saving")
        data_fits.to_pickle(fit_file)
        data_fit_metrics.to_pickle(fit_summary_file)


