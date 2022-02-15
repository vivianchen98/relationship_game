from game import *
from ipd import *
from strategies import *
from matplotlib import pyplot as plt

# class ethical(Strategy):
#     def __init__(self):
#         super().__init__()
#         self.name = "ethical"
#         self.hisPast = ""
#
#     def getAction(self, tick):
#         pass
#
#     def clone(self):
#         return ethical()
#
#     def update(self, my, his):
#         self.hisPast += his

# helper functions
def sum_2d_array(arr):
    sum = 0
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            sum += arr[i][j]
    return sum

def find_scores_index(pi_0_index):
    index = 0
    for (i,j) in pi_0_index:
        if (i,j) == (0,0):
            index = 0
        elif (i,j) == (0,1):
            index = 1
        elif (i,j) == (1,0):
            index = 2
        elif (i,j) == (1,1):
            index = 3

    return index


# ethical policy iteration algorithm
# iteratively update the utility values in the game matrix so that eventually nash equilibirum is (C,C) instead of (D,D)
def ethical_iteration(scores, actions, num_player):
    gamma = 0.5
    alpha = [[1, 0.5], [0.5, 1]]

    sum_list = []
    u1_0_list = []
    u1_1_list = []

    g = Game(scores,actions)
    g.prettyPrint()
    sum = sum_2d_array(scores)
    print("sum: ", sum)
    sum_list.append(sum)

    pi_0_index = g.getNash()
    print(pi_0_index)
    pi_0 = [(actions[i], actions[j]) for (i,j) in pi_0_index]
    print("Nash: ", pi_0)

    for i in range(100):
        # \hat{u}_i^0 = r_i(\pi^0)
        scores_index = find_scores_index(pi_0_index)
        (u0_0, u0_1) = scores[scores_index]

        # \hat{u}_i^1 = f(\hat{u}_i^0)
        epsilon = 1e-10
        u1_0 = gamma * u0_0 + (1-gamma) * alpha[0][1] * u0_1
        u1_1 = gamma * u0_1 + (1-gamma) * alpha[1][0] * u0_0
        if u1_0 < epsilon:
            u1_0 = 0
        if u1_1 < epsilon:
            u1_1 = 0
        u1_0_list.append(u1_0)
        u1_1_list.append(u1_1)

        # create new game for \hat{u}^1
        print("-----------iter{}-----------".format(i))
        scores[scores_index] = (u1_0, u1_1)
        g = Game(scores, actions)
        g.prettyPrint()
        sum = sum_2d_array(scores)
        print("sum: ", sum)
        sum_list.append(sum)

        # pi^1 = Nash(\hat{u}^1)
        pi_1_index = g.getNash()
        scores_index = find_scores_index(pi_1_index)
        print("pi_1: ", pi_1_index)

        # pi^0 = argmax_{a_i}{r_i(a_i|pi^1_{-i})}
        # pi0_0
        pi0_0_scores = [scores[scores_index-2][0], scores[scores_index][0]]
        pi0_0_index = np.argmax(pi0_0_scores)
        # pi0_1
        pi0_1_scores = [scores[scores_index-1][1], scores[scores_index][1]]
        pi0_1_index = np.argmax(pi0_1_scores)
        # form pi_0
        pi_0_index = [(pi0_0_index, pi0_1_index)]
        print("pi_0: ", pi_0_index)

    return sum_list, u1_0_list, u1_1_list


# call function here
ipd_scores =[(3,3),(0,5),(5,0),(1,1)]   # Prisoner's dilemma
sum_list, u1_0_list, u1_1_list = ethical_iteration(ipd_scores, ['C','D'], 2)

plt.plot(sum_list, label='sum of game')
plt.xlabel('iter')
plt.legend()
plt.show()

plt.plot(u1_0_list, label='u1_0')
plt.plot(u1_1_list, label='u1_1')
plt.xlabel('iter')
plt.legend()
plt.show()
