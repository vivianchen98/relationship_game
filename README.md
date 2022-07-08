# relationship_game

## Description
To design weight vector on a relationship game (RG) so that is solution is socially desirable.

## Getting Started
### Installing

### Running
#### running example: generalized traffic
* A N-player, M-action running example of traffic in 
```
trafficN.jl
```

#### Global Method: order-and-design
* Given a RG, compute a sparse w using a LP, solved by Gurobi
```
julia order_and_design.jl
```



#### Gradient Method
* New game solver: Entropy-Regularized Nash Solver in
```
game_solvers/entropy_nash_solver_general.jl
```
* Entropy-Nash Gradient Descent in
```
gradient_entropy_nash_general.jl
```
* test Entropy-Nash GD by
```
test_entropy_nash_general.jl
```
* visualize Entropy-Nash GD by
```
visualize_entropy_nash_general.jl
```

### Acknowledgments
