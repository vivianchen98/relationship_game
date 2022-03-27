from game_solvers.pure_game import *
from game_solvers.zero_sum_mixed_nash_solver import ZeroSumSolver
from game_solvers.bimatrix_mixed_nash_solver import LemkeHowsonGameSolver
from matplotlib import pyplot as plt
import random
import argparse
import itertools

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int, default=10)
# parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--epsilon', type=float, default=0.2)
# parser.add_argument('--delta', type=float, default=0.1)
# parser.add_argument('--nash_type', type=str, default='nash')
# parser.add_argument('--scores_type', type=str, default='original')
# parser.add_argument('--player_info', type=str, default='complete')
parser.add_argument('--plot', default=True, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

# helper function: to plot results
def plot_result(player_0_scores_list, player_1_scores_list, actions):
    # generate action_profiles (cartesian product of actions)
    action_profiles = list(itertools.product(actions, actions))
    action_profiles_list = [' '.join(t) for t in action_profiles]

    # player0's score figure
    player_0_scores_list = np.array(player_0_scores_list)
    plt.subplot(1,2,1)
    for i, label in enumerate(action_profiles_list):
        plt.plot(player_0_scores_list[:, i], label=label, alpha=0.8)
    plt.xlim([0, args.iter])
    plt.xlabel('iter')
    plt.ylabel('game matrix score')
    plt.title('Player 1')
    plt.legend()

    # player1's score figure
    player_1_scores_list = np.array(player_1_scores_list)
    plt.subplot(1,2,2)
    for i, label in enumerate(action_profiles_list):
        plt.plot(player_1_scores_list[:, i], label=label, alpha=0.8)
    plt.xlim([0, args.iter])
    plt.xlabel('iter')
    plt.ylabel('game matrix score')
    plt.title('Player 2')
    plt.legend()

    plt.suptitle('Strategy score updates')
    plt.show()

# ethical policy iteration algorithm
def ethical_iteration(num_player, A, B, actions, num_iter=args.iter, epsilon=args.epsilon):
    """
    Two-player Ethical Game Iteration
    Input:
        - num_player, the number of players
        - A, the game matrix for player 1
        - B, the game matrix for player 2
        - actions, the actions for all player
        - num_iter, the number of iterations
        - epsilon, randomness in epsilon-greedy
    Output:
        - player_0_scores_list
        - player_1_scores_list
        - converged, whether the game converges within the given iterations
        - info if converged:
            - converge_iter, the iter number when converged
            - pi_0_choice (deterministic)
            - pi_1 (mixed solution)
    """
    # needed parameters
    alpha = [[0, 1], [1, 0]]
    gamma = [0.1, 0.1]

    # output placeholders
    player_0_scores_list = []
    player_1_scores_list = []
    converged = False
    converge_iter, greedy_pi, nash_pi = None, None, None

    # create game object, g
    g = LemkeHowsonGameSolver(A,B)
    g.prettyPrintGame()
    pi_0, (pivots, ray_term, max_iters) = g.solve_mixed_nash()
    print("pi_0: ")
    g.prettyPrintSol(pi_0)
    #
    # print("pi_0 P1 (y): ", y)
    # print("pi_0 P2 (z): ", z)

    pi_0_support = g.support(pi_0)
    print("support: ", pi_0_support)
    pi_0_choice = [random.choice(pi_0_support_i) for pi_0_support_i in pi_0_support]

    # store original utility function
    original_g = g
    original_nash_pi_choice = pi_0_choice

    # variables to lable whether all entries in game matrix visited
    pi_0_choice_visited = []
    all_visited = False

    for i in range(num_iter):
        print("-----------iter{}-----------".format(i))

        # label whether pi_0_choice visited before
        if len(pi_0_choice_visited) < len(actions) ** 2:
            if pi_0_choice not in pi_0_choice_visited:
                pi_0_choice_visited.append(pi_0_choice)
        else:
            all_visited = True

        # \hat{u}^0_i = \Tilde{u}_i(\pi^0_i)
        u0_0 = original_g.A[tuple(pi_0_choice)]
        u0_1 = original_g.B[tuple(pi_0_choice)]

        # u_i(\pi^0_i) = g(\hat{u}^0_i)
        u1_0 = gamma[0] * u0_0 + (1-gamma[0]) * alpha[0][1] * u0_1
        u1_1 = gamma[1] * u0_1 + (1-gamma[1]) * alpha[1][0] * u0_0

        # store previous game
        g_previous = g

        # update u
        g.A[tuple(pi_0_choice)] = u1_0
        g.B[tuple(pi_0_choice)] = u1_1
        player_0_scores_list.append(g.A.flatten().tolist())
        player_1_scores_list.append(g.B.flatten().tolist())

        # create and solve the updated game: pi^1 = mixed-Nash(g)
        g.prettyPrintGame()
        pi_1, (pivots, ray_term, max_iters) = g.solve_mixed_nash()
        print("pi_1: ")
        g.prettyPrintSol(pi_1)

        # epsilon-greedy action profile: pi_0 = {(epsilon): random action profile; (1-epsilon): argmax_{a_i \in A_i} {u_i(a_i|pi^1_{-i})}}
        pi_0_choice = []
        eps = np.random.random()
        if eps < epsilon:
            pi_0_choice = [np.random.randint(0,len(actions)), np.random.randint(0,len(actions))]
            print("pi_0 (random): ", pi_0_choice)
        else:
            # compute expected utility and choose the minimizer (cost matrix)
            pi_0_choice = [np.argmin(np.dot(g.A, pi_1[1])), np.argmin(np.dot(pi_1[0], g.B))]
            print("pi_0 (greedy): ", pi_0_choice)

        # termination/convergence test
        # if g.A.all() == g_previous.A.all() and g.B.all() == g_previous.B.all() and all_visited:
        #     converged = True
        #     converge_iter = i
        #     print("CONVERGED!!!!")
        #     break

    # output convergence result
    print("\n************ Result **************")
    print("Nash sols for original game: ", original_nash_pi_choice)
    print("Nash action profile: ", (actions[original_nash_pi_choice[0]], actions[original_nash_pi_choice[1]]))
    print()
    # if converged:
    # print("converge at iter", converge_iter)
    print("* pi_0", pi_0_choice)
    print("* pi_0 profile: ", (actions[pi_0_choice[0]], actions[pi_0_choice[1]]))
    print("- pi_1 P1 (y) for " + str(actions) + ': ', pi_1[0])
    print("- pi_1 P2 (z) for " + str(actions) + ': ', pi_1[1])
    print()
    # plot result if converged and plot requested
    if args.plot:
        plot_result(player_0_scores_list, player_1_scores_list, actions)
    # else:
    #     print("does not converge for {} iterations".format(num_iter))
    #     print()

    return player_0_scores_list, player_1_scores_list, converged, (converge_iter, pi_0_choice, pi_1)  #info = (converge_iter, pi_0_indices, pi_1_indices)

# Prisoner's dilemma
ipd_scores =[(3,3),(0,5),(5,0),(1,1)]
ipd_actions = ['C','D']
# player_0_scores_list, player_1_scores_list, converged, (converge_iter, pi_0_indices, pi_1_indices) = ethical_iteration(2, ipd_scores, ipd_actions)

# Rock-scissors-paper
rsp_scores = [(0,0), (1,-1), (-1,1), (-1,1), (0,0), (1,-1), (1,-1), (-1,1), (0,0)]
rsp_actions =['R', 'S', 'P']
# player_0_scores_list, player_1_scores_list, converged, (converge_iter, pi_0_indices, pi_1_indices) = ethical_iteration(2, rsp_scores, rsp_actions)

# traffic example
# traffic_scores = [(-2,-2), (-4,-1.5), (-1.5,-4), (-3,-3)]
traffic_A = [[2,4], [1.5, 3]]
traffic_B = [[2,1.5], [4, 3]]
traffic_actions = ['Route A', 'Route B']
# player_0_scores_list, player_1_scores_list, converged, (converge_iter, pi_0_indices, pi_1_indices) = ethical_iteration(2, traffic_A, traffic_B, traffic_actions)

# traffic example (scaled up: 3 actions)
# traffic_3_scores = [(-2,-2), (-4,-1.5), (-4,-2.5),
#                     (-1.5,-4), (-3,-3), (-1.5,-2.5),
#                     (-2.5,-4), (-2.5,-1.5), (-2.5,-2.5)]
traffic_3_A = [[2, 4, 4], [1.5, 3, 1.5], [2.5, 2.5, 2.5]]
traffic_3_B = [[2, 1.5, 2.5], [4, 3, 2.5], [4, 1.5, 2.5]]
traffic_3_actions = ['Route A', 'Route B', 'Route C']
player_0_scores_list, player_1_scores_list, converged, (converge_iter, pi_0_indices, pi_1_indices) = ethical_iteration(2, traffic_3_A, traffic_3_B, traffic_3_actions)
