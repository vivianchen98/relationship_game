import pickle
import csv
import os
import argparse
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data1 = pickle.load(f)
        data2 = pickle.load(f)
    return data1, data2

# set the game and lambda by command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--game", type=str, default="congestion")
parser.add_argument("--lam", type=float, default=0.3)
args = parser.parse_args()

# load the data
J_dict_minmax, w_dict_of_lists = load_pickle("{}_minmax.pkl".format(args.game))
J_dict_downstairs, w_dict_of_lists = load_pickle("{}_downstairs.pkl".format(args.game))

# get the list of iterations
iter_list = range(max(len(J_dict_minmax[args.lam]), len(J_dict_downstairs[args.lam])))

# create dir csv if not exists
if not os.path.exists("csv"):
    os.makedirs("csv")

# export as csv
with open("csv/{}_{}.csv".format(args.game, args.lam), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["iter", "minmax", "downstairs"])
    for i in iter_list:
        if i >= len(J_dict_minmax[args.lam]):
            writer.writerow([i+1, None, J_dict_downstairs[args.lam][i]])
        elif i >= len(J_dict_downstairs[args.lam]):
            writer.writerow([i+1, J_dict_minmax[args.lam][i], None])
        else:
            writer.writerow([i+1, J_dict_minmax[args.lam][i], J_dict_downstairs[args.lam][i]])    