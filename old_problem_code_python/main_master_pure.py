from game_solvers.pure_game import *
from matplotlib import pyplot as plt
import argparse
# import wandb

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int, default=10, required=True)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--nash_type', type=str, default='nash')
parser.add_argument('--scores_type', type=str, default='original')
parser.add_argument('--player_info', type=str, default='complete')
args = parser.parse_args()

# wandb setup
# config = dict(
#     iter = args.iter,
#     gamma = args.gamma,
#     epsilon = args.epsilon
# )
# wandb.init(
#     project="ethical_game",
#     config = config,
#     mode="disabled",
# )
# wandb.run.name = '--iter ' + str(args.iter) + ' --gamma ' + str(args.gamma) + \
#                  ' --epsilon ' + str(args.epsilon) + ' --nash_type ' + args.nash_type + \
#                  ' --scores_type ' + args.scores_type
# wandb.run.save()

# helper functions
def sum_2d_array(arr):
    sum = 0
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            sum += arr[i][j]
    return sum


# ethical policy iteration algorithm
# iteratively update the utility values in the game matrix so that eventually nash equilibirum is (C,C) instead of (D,D)
def ethical_iteration(num_player, scores, actions, gamma=0.5, epsilon=0.1):
    original_scores = scores.copy()

    alpha = [[0, 1], [1, 0]]
    gamma = [0.1, 0.1]

    sum_list = []
    player_0_scores_list = []
    player_1_scores_list = []

    g = Game(scores,actions) # Game object
    g.prettyPrint()
    sum = sum_2d_array(scores)
    print("sum: ", sum)
    sum_list.append(sum)

    pi_0_indices = g.getNash()
    # pi_0 = [(actions[i], actions[j]) for (i,j) in pi_0_index]
    print("Nash: ", pi_0_indices)

    for i in range(args.iter):
        # \hat{u}_i^0 = r_i(\pi^0)
        if args.player_info == 'partial':
            # pi_empty = True
            # if len(pi_0_indices) > 0:
            #     pi_empty == False
            for pi_0_index in pi_0_indices:
                scores_index = len(actions) * pi_0_index[0] + pi_0_index[1]
                (u0_0, u0_1) = scores[scores_index]
            player_0_scores_list.append([s[0] for s in scores])
            player_1_scores_list.append([s[1] for s in scores])
        elif args.player_info == 'complete':
            u0_0 = [entry[0] for entry in scores]
            u0_1 = [entry[1] for entry in scores]
            player_0_scores_list.append(u0_0)
            player_1_scores_list.append(u0_1)

        # \hat{u}_i^1 = f(\hat{u}_i^0)
        if args.player_info == 'partial':
            # if pi_empty == True:
            #     continue
            u1_0 = gamma[0] * u0_0 + (1-gamma[0]) * alpha[0][1] * u0_1
            u1_1 = gamma[1] * u0_1 + (1-gamma[1]) * alpha[1][0] * u0_0
        elif args.player_info == 'complete':
            u1_0 = []
            u1_1 = []
            for ind in range(4):
                u1_0.append(gamma[0] * u0_0[ind] + (1-gamma[0]) * alpha[0][1] * u0_1[ind])
                u1_1.append(gamma[1] * u0_1[ind] + (1-gamma[1]) * alpha[1][0] * u0_0[ind])

        # create new game for \hat{u}^1
        print("-----------iter{}-----------".format(i))
        if args.player_info == 'partial':
            scores[scores_index] = (u1_0, u1_1)
        elif args.player_info == 'complete':
            for ind in range(4):
                scores[ind] = (u1_0[ind], u1_1[ind])

        g = Game(scores, actions)
        g.prettyPrint()
        sum = sum_2d_array(scores)
        print("sum: ", sum)
        sum_list.append(sum)

        # pi^1 = epsilon_Nash(\hat{u}^1) (epsilon-greedy Nash)
        if args.nash_type == 'epsilon_nash':
            eps = np.random.random()
            if eps < epsilon:
                pi_1_indices = [(np.random.randint(0,2), np.random.randint(0,2))]
            else:
                pi_1_indices = g.getNash()
                print("real Nash")
        elif args.nash_type == 'nash':
            pi_1_indices = g.getNash()
        print("pi_1: ", pi_1_indices)

        # pi^0 = argmax_{a_i}{r_i(a_i|pi^1_{-i})}
        pi_0_indices = []

        if args.scores_type == 'original':
            some_scores = original_scores
        elif args.scores_type == 'current':
            some_scores = scores

        for pi_1_index in pi_1_indices:
            # locate the entry for this index
            scores_index = len(actions) * pi_1_index[0] + pi_1_index[1]
            # pi0_0
            pi0_0_scores = [some_scores[scores_index-2][0], some_scores[scores_index][0]]
            pi0_0_index = np.argmax(pi0_0_scores)
            # pi0_1
            pi0_1_scores = [some_scores[scores_index-1][1], some_scores[scores_index][1]]
            pi0_1_index = np.argmax(pi0_1_scores)
            # form pi_0
            pi_0_indices.append((pi0_0_index, pi0_1_index))
        # pi_0_index = [(pi0_0_index, pi0_1_index)]
        print("pi_0: ", pi_0_indices)

    return sum_list, player_0_scores_list, player_1_scores_list


# call function here

# Prisoner's dilemma
ipd_scores =[(3,3),(0,5),(5,0),(1,1)]
ipd_actions = ['C','D']
sum_list, player_0_scores_list, player_1_scores_list = ethical_iteration(2, ipd_scores, ipd_actions, args.gamma, args.epsilon)

# Rock-scissors-paper
rsp_scores = [(0,0), (1,-1), (-1,1), (-1,1), (0,0), (1,-1), (1,-1), (-1,1), (0,0)]
rsp_actions =['R', 'S', 'P']
# sum_list, player_0_scores_list, player_1_scores_list = ethical_iteration(2, rsp_scores, rsp_actions, args.gamma, args.epsilon)


# wandb.log({'sum of scores': sum_list})

# plt.plot(sum_list, label='sum of game')
# plt.xlabel('iter')
# plt.legend()
# plt.show()

# player0's score figure
player_0_scores_list = np.array(player_0_scores_list)
plt.subplot(1,2,1)
for i, label in enumerate(['CC', 'CD', 'DC', 'DD']):
    plt.plot(player_0_scores_list[:, i], label=label)
plt.xlabel('iter')
plt.ylabel('game matrix score')
plt.title('Player 0')
plt.legend()

# player1's score figure
player_1_scores_list = np.array(player_1_scores_list)
plt.subplot(1,2,2)
for i, label in enumerate(['CC', 'CD', 'DC', 'DD']):
    plt.plot(player_1_scores_list[:, i], label=label)
plt.xlabel('iter')
plt.ylabel('game matrix score')
plt.title('Player 1')
plt.legend()

plt.suptitle('Strategy score updates')
plt.show()
