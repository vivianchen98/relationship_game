from matplotlib import pyplot as plt
import argparse
import numpy as np
from game_solvers.zero_sum_mixed_nash_solver import ZeroSumSolver
from game_solvers.bimatrix_mixed_nash_solver import LemkeHowsonGameSolver

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int, default=10, required=True)
parser.add_argument('--gamma', type=float, default=0.5)
args = parser.parse_args()

# ethical game iteration algorithm (bimatrix)
def ethical_iteration(num_player, A, B, actions, gamma=0.5):
    """
    Two-player Ethical Game Iteration
    Input:
        - N, number of players (N=2)
        - A, game matrix for P1
        - B, game matrix for P2
        - actions for each player
        - gamma, selfish factor for each player
    Output:
        - pi_0, mixed-strategy Nash equilibrium of original game G
        - pi_1, mixed_strategy Nash equilibrium of modified game G'
        - P1_scores_list, trace of game matrix A
        - P2_scores_list, trace of game matrix B
    """

    assert A.shape == B.shape, "A, B of different shape!"

    alpha = [[0, 1], [1, 0]]
    gamma = [0.5, 0.5]

    P1_scores_list = []
    P2_scores_list = []

    # construct original game G, and compute pi_0
    g = LemkeHowsonGameSolver(A,B)
    g.prettyPrint()

    y, z, (pivots, ray_term, max_iters) = g.solve_mixed_nash()
    pi_0 = (y,z)
    print("pi_0 P1 (y): ", y)
    print("pi_0 P2 (z): ", z)

    for i in range(args.iter):
        print("-----------iter{}-----------".format(i))

        # store game scores
        P1_scores_list.append(A.flatten().tolist())
        P2_scores_list.append(B.flatten().tolist())

        # U'= f(U) = \Gamma \alpha U
        A_ = gamma[0] * A + (1-gamma[0]) * alpha[0][1] * A
        B_ = gamma[1] * B + (1-gamma[1]) * alpha[1][0] * B

        # create new game G'
        g = LemkeHowsonGameSolver(A_, B_)
        g.prettyPrint()

        # pi^1 = mixed_Nash(G')
        y_, z_ , (pivots, ray_term, max_iters) = g.solve_mixed_nash()
        pi_1 = (y_, z_)
        print("pi_1 P1 (y): ", y_)
        print("pi_1 P2 (z): ", z_)

        # U = U'
        A = A_
        B = B_

    return pi_0, pi_1, P1_scores_list, P2_scores_list


# call function here

# Prisoner's dilemma
A = np.array([[3,0], [5,1]])
B = np.array([[3,5], [0,1]])
ipd_actions = ['C','D']
pi_0, pi_1, P1_scores_list, P2_scores_list = ethical_iteration(2, A, B, ipd_actions, args.gamma)

# zero-sum game example
A = np.array([[3, 0], [-1, 1]])
B = -A
actions = ['0', '1']
# pi_0, pi_1, P1_scores_list, P2_scores_list = ethical_iteration(2, A, B, ipd_actions, args.gamma)

# Rock-scissors-paper
# rsp_scores = [(0,0), (1,-1), (-1,1), (-1,1), (0,0), (1,-1), (1,-1), (-1,1), (0,0)]
# rsp_actions =['R', 'S', 'P']
# pi_0, pi_1, P1_scores_list, P2_scores_list = ethical_iteration(2, A, B, ipd_actions, args.gamma)


# player0's score figure
player_1_scores_list = np.array(P1_scores_list)
plt.subplot(1,2,1)
for i, label in enumerate(['CC', 'CD', 'DC', 'DD']):
    plt.plot(player_1_scores_list[:, i], label=label)
plt.xlabel('iter')
plt.ylabel('game matrix score')
plt.title('Player 1')
plt.legend()

# player1's score figure
player_2_scores_list = np.array(P2_scores_list)
plt.subplot(1,2,2)
for i, label in enumerate(['CC', 'CD', 'DC', 'DD']):
    plt.plot(player_2_scores_list[:, i], label=label)
plt.xlabel('iter')
plt.ylabel('game matrix score')
plt.title('Player 2')
plt.legend()

plt.suptitle('Strategy score updates')
plt.show()
