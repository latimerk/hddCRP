
import pickle


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

def get_group(subject : str) -> str:
    data = open_data()       
    
    grps = data["group_definition"];
    # grp_names = ['diverse_TH', 'diverse_HT', 'uniform_H', 'uniform_T']
    for grp_name in grp_names:
        if(subject in grps[grp_name]):
            return grp_name
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