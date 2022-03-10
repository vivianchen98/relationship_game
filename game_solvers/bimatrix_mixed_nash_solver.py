import pyomo.environ as pyo
import numpy as np

"""
Implements Lemke-Howson pivoting method for bimatrix games.
"""

class LemkeHowsonGameSolver:
    def __init__(self, A, B):
        self.A = np.array(A)
        self.B = np.array(B)
        self.max_pivots = 1000

    def solve_mixed_nash(self, epsilon=0.0):
        """
        Returns:
        - x: mixed eq strategies for player 1
        - y: mixed eq strategies for player 2
        - V1: eq game value for player 1 (Note: A is *cost* matrix for P1)
        - V2: eq game value for player 2 (Note: B is *cost* matrix for P2)
        - info: additional solver info as named tuple of
            - pivots: number of pivots performed to find solution
            - ray_term: true if method exited prematurely with a ray termination
        """

        # normalize bimatrix
        A_pos = np.add(self.A, 1.0 - np.amin(A))
        B_pos = np.add(self.B, 1.0 - np.amin(B))

        m, n = self.A.shape

        # create T
        x_factor = 1.0/(1.0-m*epsilon)
        y_factor = 1.0/(1.0-n*epsilon)

        rhs = np.block([
                [-np.ones((m,1)) * x_factor],
                [-np.ones((n,1)) * y_factor]
                ])
        lhs = np.identity(m+n)
        middle = np.block([
                [np.zeros((m,m)), -A_pos],
                [-B_pos.T,        np.zeros((n,n))]
                ])
        T = np.block([lhs, middle, rhs])

        basis = np.array([range(1,m+n+1)]).T

        r = np.argmin(B_pos.T[:, 0])
        self.pivot(T, m+r, m+n)
        basis[m+r] = m+n+1

        s = np.argmin(A_pos[:, r])
        self.pivot(T, s, m+n+m+r)
        basis[s] = m+n+m+r+1

        entering = n+m+s+1

        pivots = 0
        ray_term = False
        max_iters = False
        while (1 in basis) and (m+n+1 in basis):
                d = T[:, entering-1]
                wrong_dir = d <= 0
                wrong_dir = wrong_dir.astype(int)
                ratios = np.float64(T[:,-1]) / d
                ratios[wrong_dir] = np.inf
                t = np.argmin(ratios)
                if not all(wrong_dir):
                        self.pivot(T, t, entering-1)
                        exiting = basis[t]
                        basis[t] = entering
                        if exiting > m+n:
                                entering = exiting - m - n
                        else:
                                entering = exiting + m + n
                else:
                        ray_term = True
                        break
                pivots += 1
                if pivots >= self.max_pivots:
                        max_iters = True
                        break

        vars = np.zeros(2*(m+n))
        np.put(vars, basis-1, T[:,-1])

        u = vars[0:m]
        v = vars[m:m+n]
        x = vars[m+n:m+n+m]
        y = vars[m+n+m:m+n+m+n]

        x_sum = sum(x)
        y_sum = sum(y)

        # import pdb; pdb.set_trace()

        x_normalized = (np.float64(x) / (x_sum * x_factor)) + epsilon
        y_normalized = (np.float64(y) / (y_sum * y_factor)) + epsilon

        return x_normalized, y_normalized, (pivots, ray_term, max_iters)

    def pivot(self, T, row, col):
        pivot = T[row, :] / T[row, col]
        T -= np.matmul(T[:, col].reshape(-1,1), pivot.reshape(1,pivot.size))
        T[row, :] = pivot.T


# test example
A = [[3,2], [2,3]]
B = [[1,2], [3,1]]
# expected result
# x = (2/3, 1/3)
# y = (1/2, 1/2)

solver = LemkeHowsonGameSolver(A, B)
x_normalized, y_normalized, (pivots, ray_term, max_iters) = solver.solve_mixed_nash()
print("x: ", x_normalized)
print("y: ", y_normalized)
