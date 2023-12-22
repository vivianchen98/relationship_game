# relationship_game

To design a weight vector on a relationship game so that the quantal response equilibrium minimizes social cost.

### Relationship game examples
* Two congestion game examples `congestion()` and `bee_queen()` in
```
examples.jl
```

### Game Solvers and Inverse algorithms
* Game solver: Quantal Response Equilibrium (QRE) `solve_entropy_nash_jump` in
```
game_solvers/entropy_nash_solver_jump.jl
```
* Projected Gradient Descent algorithms `ProjectedGradientMinMax` and `ProjectedGradientDownstairs` that solves the *Min-Max* and Min-Min problems in
```
game_solvers/gradient_entropy_nash_jump_new.jl
```

### Experiment
* run experiments in
```
test.jl
```
