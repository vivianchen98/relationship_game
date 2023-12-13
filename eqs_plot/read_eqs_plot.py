import pickle
import csv
import os
import argparse
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data1 = pickle.load(f)
        data2 = pickle.load(f)
        data3 = pickle.load(f)
    return data1, data2, data3

# set the game and lambda by command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--game", type=str, default="bee_queen")
parser.add_argument("--axis", type=int, default=3)
parser.add_argument("--lam", type=float, default=1.0)
args = parser.parse_args()

# load the data
dataset, dataset_max, dataset_min = load_pickle("{}/w{}_lambda{}.pkl".format(args.game, args.axis, args.lam))

# save to csv
if not os.path.exists("csv"):
    os.makedirs("csv")

# with open("csv/{}_w{}_lambda{}.csv".format(args.game, args.axis, args.lam), "w") as f:
#     writer = csv.writer(f)
#     writer.writerow(["w_axis", "data"])
#     for i in range(len(dataset)):
#         writer.writerow([dataset[i][0], dataset[i][1]])

# with open("csv/{}_w{}_lambda{}_max.csv".format(args.game, args.axis, args.lam), "w") as f:
#     writer = csv.writer(f)
#     writer.writerow(["w_axis", "data"])
#     for i in range(len(dataset_max)):
#         writer.writerow([dataset_max[i][0], dataset_max[i][1]])

# with open("csv/{}_w{}_lambda{}_min.csv".format(args.game, args.axis, args.lam), "w") as f:
#     writer = csv.writer(f)
#     writer.writerow(["w_axis", "data"])
#     for i in range(len(dataset_min)):
#         writer.writerow([dataset_min[i][0], dataset_min[i][1]])

with open("csv/{}_w{}_lambda{}_max_equals_min.csv".format(args.game, args.axis, args.lam), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["w_axis", "max_equals_min"])
    for i in range(len(dataset_max)):
        if dataset_max[i][1] == dataset_min[i][1]:
            writer.writerow([dataset_max[i][0], dataset_max[i][1]])