from matplotlib import pyplot as plt
import argparse
import numpy as np
from game_solvers.zero_sum_mixed_nash_solver import ZeroSumSolver
from game_solvers.bimatrix_mixed_nash_solver import LemkeHowsonGameSolver

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int, default=10, required=True)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--delta', type=float, default=0.1)
parser.add_argument('--plot', default=True, action=argparse.BooleanOptionalAction)
args = parser.parse_args()


# helper function: to plot results
def plot_result(player_0_scores_list, player_1_scores_list):
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

# ethical game iteration algorithm (bimatrix)
def ethical_iteration(num_player, A, B, actions, num_iter=args.iter, delta=args.delta):
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
    gamma = [0.1, 0.1]

    P1_scores_list = []
    P2_scores_list = []
    converged = False
    converge_iter, greedy_pi, nash_pi = None, None, None

    # construct original game G, and compute pi_0
    g = LemkeHowsonGameSolver(A,B)
    g.prettyPrint()

    y, z, (pivots, ray_term, max_iters) = g.solve_mixed_nash()
    pi_0 = (y,z)
    print("pi_0 P1 (y): ", y)
    print("pi_0 P2 (z): ", z)

    for i in range(num_iter):
        print("-----------iter{}-----------".format(i))

        # store game scores
        P1_scores_list.append(A.flatten().tolist())
        P2_scores_list.append(B.flatten().tolist())

        # U'= f(U) = \Gamma \alpha U
        A_ = gamma[0] * A + (1-gamma[0]) * alpha[0][1] * B
        B_ = gamma[1] * B + (1-gamma[1]) * alpha[1][0] * A

        # create new game G'
        g_ = LemkeHowsonGameSolver(A_, B_)
        g_.prettyPrint()

        # calculate L2-norm of difference between last and current game matrices
        g_diff_A, g_diff_B = A - A_, B - B_
        g_diff_A_norm, g_diff_B_norm = np.linalg.norm(g_diff_A), np.linalg.norm(g_diff_B)

        if g_diff_A_norm < delta and g_diff_B_norm < delta:
            converged = True
            converge_iter = i
            greedy_pi = (np.argmin(A_) // 2, np.argmin(B_) % 2) # greedy: minimize cost(positive)
            y_, z_ , (pivots, ray_term, max_iters) = g_.solve_mixed_nash()
            break
        else:
            print("A norm: ", g_diff_A_norm)
            print("B norm: ", g_diff_B_norm)

        # U = U'
        A = A_
        B = B_

    # output convergence result
    print("\n************ Result **************")
    print("Nash sols for original game: ", pi_0)
    print()
    if converged:
        print("converge at iter", converge_iter)
        print("* greedy_pi", greedy_pi)
        print("- nash_pi P1 (y): ", y_)
        print("- nash_pi P2 (z): ", z_)
        print()
    else:
        print("does not converge (L2-norm of game < {}) for {} iterations".format(delta, num_iter))
        print()

    # plot result if needed
    if args.plot:
        plot_result(P1_scores_list, P2_scores_list)

    return P1_scores_list, P2_scores_list


# call function here

# Prisoner's dilemma
A = np.array([[1,3], [0,2]])
B = np.array([[1,0], [3,2]])
ipd_actions = ['C','D']
# P1_scores_list, P2_scores_list = ethical_iteration(2, A, B, ipd_actions, args.iter, args.delta)

# traffic example
traffic_A = np.array([[2,4], [1.5,3]])
traffic_B = np.array([[2,1.5], [4,3]])
traffic_actions = ['Route A', 'Route B']
P1_scores_list, P2_scores_list = ethical_iteration(2, traffic_A, traffic_B, traffic_actions, args.iter, args.delta)

# zero-sum game example
A = np.array([[3, 0], [-1, 1]])
B = -A
actions = ['0', '1']
# pi_0, pi_1, P1_scores_list, P2_scores_list = ethical_iteration(2, A, B, ipd_actions, args.gamma)

# Rock-scissors-paper
rsp_scores = [(0,0), (1,-1), (-1,1), (-1,1), (0,0), (1,-1), (1,-1), (-1,1), (0,0)]
rsp_actions =['R', 'S', 'P']
# pi_0, pi_1, P1_scores_list, P2_scores_list = ethical_iteration(2, A, B, ipd_actions, args.gamma)
