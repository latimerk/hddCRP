
import pickle
import numpy as np
import pandas as pd
import os.path
grp_names = ['diverse_TH', 'diverse_HT', 'uniform_H', 'uniform_T']
grp_names_all = ['diverse_TH', 'diverse_HT', 'uniform_H', 'uniform_T', 'diverse', 'uniform']

def open_data():
    data_filename = 'data/Data_turns_all_by_session.pkl';
    with open(data_filename, 'rb') as data_file:
        data_file = pickle.load(data_file)
    return data_file

def get_subjects(grp_name : str = None) -> list:
    data = open_data()        
    if(grp_name is None):
        subjects = list(data["data"].keys())
    else:
        subjects = data["group_definition"][grp_name];

    return subjects

def get_group(subject : str, full : bool = True) -> str:
    data = open_data()       
    
    grps = data["group_definition"];
    # grp_names = ['diverse_TH', 'diverse_HT', 'uniform_H', 'uniform_T']
    for grp_name in grp_names:
        if(subject in grps[grp_name]):
            if(full):
                grp_name_c = grp_name
            else:
                grp_name_c = grp_name.split("_")[0]
            return grp_name_c
    return None

def get_data(subject : str, return_session_labels : bool = False) -> list | tuple[list,list]:
    data = open_data()       
    sequences_0 = data["data"][subject]["data"]
    session_types_0 = data["data"][subject]["task"]

    return sequences_0, session_types_0

def get_phase1_(subject : str, offset : int, return_session_labels : bool = False) -> tuple[list,list]:
    seqs, session_types = get_data(subject, return_session_labels = True)
    idx = session_types.index('C')
    if(offset is None):
        ss = 0
        dd = 2
    else:
        ss = (idx + offset) % 2
        dd = 1
    if(return_session_labels):
        return seqs[ss:idx:2], session_types[ss:idx:dd]
    else:
        return seqs[ss:idx:2]

def get_phase1_a(subject : str, return_session_labels : bool = False) -> list | tuple[list, list]:
    return get_phase1_(subject, 0, return_session_labels)
def get_phase1_b(subject : str, return_session_labels : bool = False) -> list | tuple[list, list]:
    return get_phase1_(subject, 1, return_session_labels)
def get_phase1(subject : str, return_session_labels : bool = False) -> list | tuple[list, list]:
    return get_phase1_(subject, None, return_session_labels)

def get_phase2(subject : str, return_session_labels : bool = False) -> list | tuple[list,list]:
    seqs, session_types = get_data(subject, return_session_labels = True)
    idx = session_types.index('C')
    if(return_session_labels):
        return seqs[idx:], session_types[:idx]
    else:
        return seqs[idx:]
    


def load_raw_with_reward_phase2(subject : str, remove_last_trial : bool = True, use_turns : bool = True, use_recoded_turns : bool = False) -> [list,list]:
    """ Loads the trials performed by a single rat during the plus maze phase from the raw csv files.
        Includes the co-registered reward information.

    Args:
      subject: The name of the animal (e.g., A1)
      remove_last_trial: removes last trial that I see in each session the csv files if true. This was needed to replicate what I saw in the processsed data file.

    Returns:
      sequences, rewards: Each return value is a list of the J sessions that the rat performed.
        Each entry is a list of the actions (0=left turn, 1=straight, 2=right turn) and reward information (0=action not rewarded, 1=action rewarded) for each session.
        For session jj, sequences[jj] is the list of actions and rewards[jj] is the rewards: therefore len(sequences[jj]) = len(rewards[jj]).
    """

    file_name = f"data/raw/{subject}_C.csv";

    if(not os.path.exists(file_name)):
        # NOTE: check to see if you have the data in the correct spots
        raise RuntimeError(f"Raw data not found. Expected data file in path: {file_name}")

    df = pd.read_csv(file_name, index_col=0)
    df.sort_values(["session", "on_time"], inplace=True)
    df = df[["session", "Rewarded", "well_id", "on_time"]]
    df["session_id"] = df.groupby(df["session"]).ngroup()
    df["trial_number"] = df.groupby("session_id")["well_id"].transform(lambda rr : (rr != rr.shift()).cumsum() - 1)

    def get_first_well_entry(pp):
        xx = pp.iloc[0]
        xx["Rewarded"] = pd.Series.any(pp["Rewarded"])
        return xx
    df = df.groupby(["session","trial_number"],as_index=False).apply(get_first_well_entry)
    df.sort_values(["session", "trial_number"], inplace=True)


    def turns(well_id):
        well_id -= 1
        ss = (well_id == (well_id.shift() + 2) % 4);
        ll = (well_id == (well_id.shift() + 1) % 4);
        rr = (well_id == (well_id.shift() - 1) % 4);
        well_id = well_id.astype(str)
        well_id[:] = "None"
        well_id[ss] = "Straight"
        well_id[ll] = "Left"
        well_id[rr] = "Right"
        return well_id
    def recode_turns(turn_id):
        sw = (turn_id == (turn_id.shift() + 2) % 3)
        rp = (turn_id == (turn_id.shift() + 3) % 3)
        st = (turn_id == 1)
        na = (turn_id.shift(fill_value=pd.NA) == pd.NA)
        turn_id[sw] = 0
        turn_id[rp] = 2
        turn_id[st] = 1
        turn_id[na] = pd.NA
        return turn_id

    df["turn"]     = df.groupby(["session"],as_index=False)["well_id"].transform(turns)
    df["turn_idx"] = df["turn"].map({"None" : pd.NA, "Straight" : 1, "Left" : 0, "Right" : 2})

    df["recoded_turn_idx"]     = df.groupby(["session"],as_index=False)["turn_idx"].transform(recode_turns)


    grouped = df.groupby(["session_id"])

    seqs = []
    reward = []
    session_labels = []
    for session, group in grouped:
        session_labels.append(group["session"].iloc[0])


        if(use_recoded_turns):
            tts = group["recoded_turn_idx"]
        elif(use_turns):
            tts = group["turn_idx"]
        else:
            tts = group["well_id"]

        rrs = group["Rewarded"]
        rrs = rrs[pd.notna(tts)].values
        tts = tts[pd.notna(tts)].values
        if(remove_last_trial):
            tts = tts[:-1] # NOTE: needed to cut out the last one to match the turn data structure
            rrs = rrs[:-1]
        seqs.append(list(tts))
        reward.append(list(rrs))
    return seqs, reward
