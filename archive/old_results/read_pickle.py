import pickle

def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data1 = pickle.load(f)
        data2 = pickle.load(f)
    return data1, data2

J_dict_of_lists, w_dict_of_lists = load_pickle("congestion.pkl")