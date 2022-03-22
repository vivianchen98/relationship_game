from game_solvers.pure_game import *
from matplotlib import pyplot as plt
import argparse

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.5)
# parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--delta', type=float, default=0.1)
parser.add_argument('--nash_type', type=str, default='nash')
# parser.add_argument('--scores_type', type=str, default='original')
parser.add_argument('--player_info', type=str, default='complete')
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

# ethical policy iteration algorithm
def ethical_iteration(num_player, scores, actions, num_iter=args.iter, delta=args.delta):
    """
    Two-player Ethical Game Iteration
    Input:
        - num_player, the number of players
        - scores, the game matrix
        - actions, the actions for all players
        - num_iter, the number of iterations
        - delta, the convergence threshold
    Output:
        - player_0_scores_list
        - player_1_scores_list
        - converged, whether the game converges within the given iterations
        - info if converged:
            - converge_iter, the iter number when converged
            - greedy_pi
            - nash_pi
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
    g = Game(scores,actions)
    g.prettyPrint()

    original_nash_pi = g.getNash()

    for i in range(num_iter):
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

        # create new game g_ for \hat{u}^1
        print("-----------iter{}-----------".format(i))

        if args.player_info == 'partial':
            scores[scores_index] = (u1_0, u1_1)
        elif args.player_info == 'complete':
            for ind in range(4):
                scores[ind] = (u1_0[ind], u1_1[ind])

        g_ = Game(scores, actions)
        g_.prettyPrint()


        # calculate L2-norm of difference between last and current game matrices
        g_diff_x, g_diff_y = g.scores["x"] - g_.scores["x"], g.scores["y"] - g_.scores["y"]
        g_diff_x_norm, g_diff_y_norm = np.linalg.norm(g_diff_x), np.linalg.norm(g_diff_y)

        if g_diff_x_norm < delta and g_diff_y_norm < delta:
            converged = True
            converge_iter = i
            greedy_pi = (np.argmax(g_.scores["x"]) // 2, np.argmax(g_.scores["y"]) % 2) # greedy: maximize utility (regardless of sign)
            nash_pi = g_.getNash()
            break
        else:
            print("x norm: ", g_diff_x_norm)
            print("y norm: ", g_diff_y_norm)

        # update game matrix
        g = g_

    # output convergence result
    print("\n************ Result **************")
    print("Nash sols for original game: ", original_nash_pi)
    for pi in original_nash_pi:
        print("Nash action profile: ", (actions[pi[0]], actions[pi[1]]))
    print()
    if converged:
        print("converge at iter", converge_iter)
        print("* greedy_pi", greedy_pi)
        print("* Greedy action profile: ", (actions[greedy_pi[0]], actions[greedy_pi[1]]))
        print("- nash_pi", nash_pi)
        for pi in nash_pi:
            print("- Nash action profile: ", (actions[pi[0]], actions[pi[1]]))
        print()
    else:
        print("does not converge (L2-norm of game < {}) for {} iterations".format(delta, num_iter))
        print()

    # plot result if needed
    if args.plot:
        plot_result(player_0_scores_list, player_1_scores_list)

    return player_0_scores_list, player_1_scores_list, converged, (converge_iter, greedy_pi, nash_pi)  #info = (converge_iter, greedy_pi, nash_pi)

# Prisoner's dilemma
ipd_scores =[(3,3),(0,5),(5,0),(1,1)]
ipd_actions = ['C','D']
# player_0_scores_list, player_1_scores_list, converged, (converge_iter, greedy_pi, nash_pi) = ethical_iteration(2, ipd_scores, ipd_actions, args.iter, args.delta)

# Rock-scissors-paper
rsp_scores = [(0,0), (1,-1), (-1,1), (-1,1), (0,0), (1,-1), (1,-1), (-1,1), (0,0)]
rsp_actions =['R', 'S', 'P']
# player_0_scores_list, player_1_scores_list, converged, (converge_iter, greedy_pi, nash_pi) = ethical_iteration(2, rsp_scores, rsp_actions, args.iter, args.delta)

# traffic example
traffic_scores = [(-2,-2), (-4,-1.5), (-1.5,-4), (-3,-3)]
traffic_actions = ['Route A', 'Route B']
player_0_scores_list, player_1_scores_list, converged, (converge_iter, greedy_pi, nash_pi) = ethical_iteration(2, traffic_scores, traffic_actions, args.iter, args.delta)
