import pyomo.environ as pyo
import numpy as np

"""
A zero-sum solver that casts the game as linear program.
"""

class ZeroSumSolver:
    def __init__(self, game_matrix):
        self.game_matrix = np.array(game_matrix)

    def solve_mixed_nash(self):
        y, V_y = self.solve_mixed_security_strategy(self.game_matrix)
        z, V_z = self.solve_mixed_security_strategy(-np.transpose(self.game_matrix))
        return y, z

    def solve_mixed_security_strategy(self, game_matrix):
        offset = -np.amin(game_matrix) + 1

        A = np.add(game_matrix, offset)
        sol_x, sol_V = self.solve_simplex_lp(A)
        x = np.divide(sol_x, sol_V)
        V = 1 / sol_V - offset
        return x, V

    def solve_simplex_lp(self, A):
        model = pyo.ConcreteModel()

        # index sets
        m, n = A.shape
        model.I = pyo.RangeSet(1,n)
        model.J = pyo.RangeSet(1,m)

        # Parameter: A
        A_set = {}
        for i in model.I:
            for j in model.J:
                A_set[i,j] = np.transpose(A)[i-1][j-1]
        model.A = pyo.Param(model.I, model.J, initialize=A_set, default=0)

        # Variable: x >= 0
        model.x = pyo.Var([1,m], domain=pyo.NonNegativeReals)

        # Constraint: A^T*x <= 1
        model.constr = pyo.ConstraintList()
        for i in model.I:
            model.constr.add(sum([model.A[i,j] * model.x[j] for j in model.J]) <= 1)

        # Objective: max x^T 1 = max sum(x)
        model.OBJ = pyo.Objective(rule=pyo.summation(model.x), sense=pyo.maximize)

        # Solve the optimization problem
        opt = pyo.SolverFactory('gurobi')
        opt.solve(model)


        x = [model.x[j]() for j in model.J]
        return x, model.OBJ()

    def solve_print(self):
        y, z = solver.solve_mixed_nash()
        print("mixed nash for P1 (y): ", y)
        print("mixed nash for P2 (z): ", z)

game_matrix = [[3, 0], [-1, 1]]
solver = ZeroSumSolver(game_matrix)
solver.solve_print()
