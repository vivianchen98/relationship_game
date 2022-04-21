from game_solvers.pure_game import *
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
def ethical_iteration(num_player, scores, actions, num_iter=args.iter, epsilon=args.epsilon):
    """
    Two-player Ethical Game Iteration
    Input:
        - num_player, the number of players
        - scores, the game matrix
        - actions, the actions for all players
        - num_iter, the number of iterations
        - epsilon, the randomness probability in epsilon-greedy
    Output:
        - player_0_scores_list
        - player_1_scores_list
        - converged, whether the game converges within the given iterations
        - info if converged:
            - converge_iter, the iter number when converged
            - pi_0_indices
            - pi_1_indices
    """
    # needed parameters
    alpha = [[0, 1], [1, 0]]
    gamma = [0.4, 0.4]

    # output placeholders
    player_0_scores_list = []
    player_1_scores_list = []
    converged = False
    converge_iter, greedy_pi, nash_pi = None, None, None

    # create game object, g
    g = Game(scores, actions)
    g.prettyPrint()
    pi_0_indices = g.getNash()
    print("pi_0: ", pi_0_indices)

    # store original utility function
    original_g = g
    original_nash_pi = pi_0_indices

    for i in range(num_iter):
        print("-----------iter{}-----------".format(i))

        player_0_scores_list.append(g.scores["x"].flatten().tolist())
        player_1_scores_list.append(g.scores["y"].flatten().tolist())

        # \hat{u}^0_i = \Tilde{u}_i(\pi^0_i)
        for pi_0_index in pi_0_indices:
            u0_0, u0_1 = original_g.scores["x"][pi_0_index], original_g.scores["y"][pi_0_index]

        # u_i(\pi^0_i) = g(\hat{u}^0_i)
        u1_0 = gamma[0] * u0_0 + (1-gamma[0]) * alpha[0][1] * u0_1
        u1_1 = gamma[1] * u0_1 + (1-gamma[1]) * alpha[1][0] * u0_0


        # store previous game
        g_previous = g

        # update u
        g.scores["x"][pi_0_index] = u1_0
        g.scores["y"][pi_0_index] = u1_1
        # import pdb; pdb.set_trace()


        # create and solve the updated game: pi^1 = Nash(g)
        g.prettyPrint()
        pi_1_indices = g.getNash()
        print("pi_1: ", pi_1_indices)

        # epsilon-greedy action profile: pi_0 = {(epsilon): random action profile; (1-epsilon): argmax_{a_i \in A_i} {u_i(a_i|pi^1_{-i})}}
        pi_0_indices = []
        eps = np.random.random()
        if eps < epsilon:
            pi_0_index = (np.random.randint(0,len(actions)), np.random.randint(0,len(actions)))
            pi_0_indices.append(pi_0_index)
        else:
            for pi_1_index in pi_1_indices:
                pi_0_index = (np.argmax(g.scores["x"][pi_1_index[1]]), np.argmax(g.scores["y"][pi_1_index[0]]))
                pi_0_indices.append(pi_0_index)
        print("pi_0: ", pi_0_indices)

        # termination/convergence test
        # if g.scores["x"].all() == g_previous.scores["x"].all() and g.scores["y"].all() == g_previous.scores["y"].all() \
        #     and len(pi_0_indices) > 0 and len(pi_1_indices) > 0 and pi_0_indices == pi_1_indices \
        #     and eps >= epsilon:
        #     converged = True
        #     converge_iter = i
        #     print("CONVERGED!!!!")
        #     break

    # output convergence result
    print("\n************ Result **************")
    print("Nash sols for original game: ", original_nash_pi)
    for pi in original_nash_pi:
        print("Nash action profile: ", (actions[pi[0]], actions[pi[1]]))
    print()
    # if converged:
    print("converge at iter", converge_iter)
    print("* pi_0", pi_0_indices)
    print("* pi_0 profile: ", [(actions[pi_0_index[0]], actions[pi_0_index[1]]) for pi_0_index in pi_0_indices])
    print("- pi_1", pi_1_indices)
    print("- pi_1 profile: ", [(actions[pi_1_index[0]], actions[pi_1_index[1]]) for pi_1_index in pi_1_indices])
    print()
    # plot if asked
    if args.plot:
        plot_result(player_0_scores_list, player_1_scores_list, actions)
    # else:
    #     print("does not converge for {} iterations".format(num_iter))
    #     print()

    # plot result if needed
    # if args.plot:
    #     plot_result(player_0_scores_list, player_1_scores_list)

    return player_0_scores_list, player_1_scores_list, converged, (converge_iter, pi_0_indices, pi_1_indices)  #info = (converge_iter, pi_0_indices, pi_1_indices)

# Prisoner's dilemma
ipd_scores =[(3,3),(0,5),(5,0),(1,1)]
ipd_actions = ['C','D']
# player_0_scores_list, player_1_scores_list, converged, (converge_iter, pi_0_indices, pi_1_indices) = ethical_iteration(2, ipd_scores, ipd_actions)

# Rock-scissors-paper
rsp_scores = [(0,0), (1,-1), (-1,1), (-1,1), (0,0), (1,-1), (1,-1), (-1,1), (0,0)]
rsp_actions =['R', 'S', 'P']
# player_0_scores_list, player_1_scores_list, converged, (converge_iter, pi_0_indices, pi_1_indices) = ethical_iteration(2, rsp_scores, rsp_actions)

# traffic example
traffic_scores = [(-2,-2), (-4,-1.5), (-1.5,-4), (-3,-3)]
traffic_actions = ['Route A', 'Route B']
# player_0_scores_list, player_1_scores_list, converged, (converge_iter, pi_0_indices, pi_1_indices) = ethical_iteration(2, traffic_scores, traffic_actions)

# traffic example (scaled up: 3 actions)
traffic_3_scores = [(-2,-2), (-4,-1.5), (-4,-2.5), (-1.5,-4), (-3,-3), (-1.5,-2.5), (-2.5,-4), (-2.5,-1.5), (-2.5,-2.5)]
traffic_3_actions = ['Route A', 'Route B', 'Route C']
player_0_scores_list, player_1_scores_list, converged, (converge_iter, pi_0_indices, pi_1_indices) = ethical_iteration(2, traffic_3_scores, traffic_3_actions)
