from game import *
from matplotlib import pyplot as plt

# helper functions
def sum_2d_array(arr):
    sum = 0
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            sum += arr[i][j]
    return sum

def find_scores_index(pi_index):
    if pi_index == (0,0):
        index = 0
    elif pi_index == (0,1):
        index = 1
    elif pi_index == (1,0):
        index = 2
    elif pi_index == (1,1):
        index = 3

    return index


# ethical policy iteration algorithm
# iteratively update the utility values in the game matrix so that eventually nash equilibirum is (C,C) instead of (D,D)
def ethical_iteration(num_player, scores, actions, gamma=0.5, epsilon=0.1):
    original_scores = scores.copy()

    alpha = [[0, 1], [1, 0]]

    sum_list = []
    u1_0_list = []
    u1_1_list = []

    g = Game(scores,actions) # Game object
    g.prettyPrint()
    sum = sum_2d_array(scores)
    print("sum: ", sum)
    sum_list.append(sum)

    pi_0_indices = g.getNash()
    # pi_0 = [(actions[i], actions[j]) for (i,j) in pi_0_index]
    print("Nash: ", pi_0_indices)

    for i in range(100):
        # \hat{u}_i^0 = r_i(\pi^0)
        u0_0 = [entry[0] for entry in scores]
        u0_1 = [entry[1] for entry in scores]

        # \hat{u}_i^1 = f(\hat{u}_i^0)
        u1_0 = []
        u1_1 = []
        for ind in range(4):
            u1_0.append(gamma * u0_0[ind] + (1-gamma) * alpha[0][1] * u0_1[ind])
            u1_1.append(gamma * u0_1[ind] + (1-gamma) * alpha[1][0] * u0_0[ind])

        # create new game for \hat{u}^1
        print("-----------iter{}-----------".format(i))
        for ind in range(4):
            scores[ind] = (u1_0[ind], u1_1[ind])
        g = Game(scores, actions)
        g.prettyPrint()
        sum = sum_2d_array(scores)
        print("sum: ", sum)
        sum_list.append(sum)

        # pi^1 = epsilon_Nash(\hat{u}^1) (epsilon-greedy Nash)
        pi_1_indices = g.getNash()
        print("pi_1: ", pi_1_indices)

        # pi^0 = argmax_{a_i}{r_i(a_i|pi^1_{-i})}
        pi_0_indices = []
        for pi_1_index in pi_1_indices:
            # locate the entry for this index
            scores_index = find_scores_index(pi_1_index)
            # pi0_0
            pi0_0_scores = [original_scores[scores_index-2][0], original_scores[scores_index][0]]
            pi0_0_index = np.argmax(pi0_0_scores)
            # pi0_1
            pi0_1_scores = [original_scores[scores_index-1][1], original_scores[scores_index][1]]
            pi0_1_index = np.argmax(pi0_1_scores)
            # form pi_0
            pi_0_indices.append((pi0_0_index, pi0_1_index))
        # pi_0_index = [(pi0_0_index, pi0_1_index)]
        print("pi_0: ", pi_0_indices)

    return sum_list, u1_0_list, u1_1_list


# call function here
ipd_scores =[(3,3),(0,5),(5,0),(1,1)]   # Prisoner's dilemma
sum_list, u1_0_list, u1_1_list = ethical_iteration(2, ipd_scores, ['C','D'], 0.5, 0.1)

# plt.plot(sum_list, label='sum of game')
# plt.xlabel('iter')
# plt.legend()
# plt.show()
#
# plt.plot(u1_0_list, label='u1_0')
# plt.plot(u1_1_list, label='u1_1')
# plt.xlabel('iter')
# plt.legend()
# plt.show()
